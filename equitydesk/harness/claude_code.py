"""A backend that runs every call through the Claude Code CLI already logged in on this machine (headless
`claude -p`), so calls are covered by the Claude subscription instead of a metered API key.

* structured(): `--json-schema` returns output validated against the pydantic schema.
* web_search(): only the WebSearch and WebFetch tools are available; the run is streamed (`stream-json`) so
  the URLs those tools actually returned are known, and the brief keeps only sources among them (the same
  provenance rule as the OpenAI path: a source the tools never returned is dropped).
* Each call runs in an empty working folder with no project settings, no MCP servers and no saved session.
"""

from __future__ import annotations

import glob
import json
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

from langchain_core.messages import BaseMessage

from equitydesk.config import ROOT, ModelCfg


class ClaudeCodeError(RuntimeError):
    pass


class NotLoggedIn(ClaudeCodeError):
    pass


def find_cli(explicit: str | None = None) -> str | None:
    """`claude` on PATH, else the CLI bundled with the Claude desktop app (newest version first)."""
    if explicit and Path(explicit).exists():
        return explicit
    on_path = shutil.which("claude")
    if on_path:
        return on_path
    local = os.environ.get("LOCALAPPDATA", "")
    found = glob.glob(os.path.join(local, "Claude-Data", "*", "claude-code", "*", "*", "claude.exe"))
    found += glob.glob(os.path.join(os.path.expanduser("~"), ".local", "bin", "claude*"))
    return max(found, key=os.path.getmtime) if found else None


WORKDIR = ROOT / "data" / "claude_work"
URL_RE = re.compile(r"https?://[^\s\"'<>)\]]+")
SOURCE_LINE = re.compile(r"^\s*\[(S\d+)\]\s*(.*?)\s*[—\-–:]\s*(https?://\S+)\s*$", re.M)


class ClaudeCodeBackend:
    def __init__(self, cfg: ModelCfg):
        self.cfg = cfg
        self.name = f"claude-code/{cfg.name}"
        self.exe = find_cli(cfg.base_url)
        if not self.exe:
            from equitydesk.harness.llm import ConfigError
            raise ConfigError("Claude Code CLI not found. Install Claude Code, or set the path in settings "
                              "(base_url of the claude_code role).")

    # -- plumbing -------------------------------------------------------------------------------
    def _run(self, system: str, prompt: str, extra: list[str], timeout: float) -> str:
        WORKDIR.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile("w", suffix=".md", delete=False, encoding="utf-8", dir=WORKDIR) as f:
            f.write(system)
            sys_file = f.name
        cmd = [self.exe, "-p", "--model", self.cfg.name, "--system-prompt-file", sys_file,
               "--no-session-persistence", "--setting-sources", "", "--strict-mcp-config", *extra]
        try:
            r = subprocess.run(cmd, input=prompt, capture_output=True, text=True, encoding="utf-8", timeout=timeout,
                               cwd=WORKDIR, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        finally:
            os.unlink(sys_file)
        if "Not logged in" in (r.stdout + r.stderr):
            raise NotLoggedIn("Claude Code is not logged in on this machine. Open a terminal and run "
                              f"`\"{self.exe}\" auth login` once, then restart the monitor.")
        if r.returncode != 0 and not r.stdout.strip():
            raise ClaudeCodeError(f"claude exited {r.returncode}: {r.stderr.strip()[:300]}")
        return r.stdout

    @staticmethod
    def _usage(meta: dict, model: str):
        from equitydesk.harness.llm import Usage
        u = meta.get("usage") or {}
        return Usage(input_tokens=int(u.get("input_tokens") or 0) + int(u.get("cache_read_input_tokens") or 0)
                     + int(u.get("cache_creation_input_tokens") or 0),
                     output_tokens=int(u.get("output_tokens") or 0),
                     cache_read_tokens=int(u.get("cache_read_input_tokens") or 0), model=model,
                     search_calls=int((u.get("server_tool_use") or {}).get("web_search_requests") or 0))

    @staticmethod
    def _split(messages: list[BaseMessage]) -> tuple[str, str]:
        system = "\n\n".join(m.content for m in messages if m.type == "system")
        user = "\n\n".join(str(m.content) for m in messages if m.type != "system")
        return system, user

    # -- the Backend protocol -------------------------------------------------------------------
    def structured(self, schema, messages):
        system, user = self._split(messages)
        out = self._run(system, user, ["--output-format", "json", "--tools", "",
                                       "--json-schema", json.dumps(schema.model_json_schema())],
                        self.cfg.timeout_s * 3)
        meta = json.loads(out)
        usage = self._usage(meta, self.name)
        if meta.get("is_error"):
            return None, f"claude: {meta.get('result') or meta.get('terminal_reason')}"[:500], usage
        data = meta.get("structured_output")
        if data is None:
            text = (meta.get("result") or "").strip()
            m = re.search(r"\{.*\}", text, re.S)
            data = json.loads(m.group(0)) if m else None
        if data is None:
            return None, "claude returned no structured output", usage
        try:
            return schema.model_validate(data), None, usage
        except Exception as e:
            return None, f"{type(e).__name__}: {e}"[:1500], usage

    def chat(self, messages):
        system, user = self._split(messages)
        meta = json.loads(self._run(system, user, ["--output-format", "json", "--tools", ""], self.cfg.timeout_s * 3))
        return meta.get("result") or "", self._usage(meta, self.name)

    def web_search(self, system: str, prompt: str, context_size: str = "medium"):
        """Search with Claude Code's own tools. The brief cites sources as [S<n>] and ends with a source list;
        only URLs the tools actually returned survive."""
        system = (system + "\n\nCite every claim inline with a marker like [S1]. End with a section 'Sources:' that "
                  "lists each marker on its own line as `[S1] Title — URL`, using only URLs you found with the tools.")
        out = self._run(system, prompt, ["--output-format", "stream-json", "--verbose", "--tools", "WebSearch,WebFetch",
                                         "--allowedTools", "WebSearch", "WebFetch", "--permission-mode", "dontAsk"],
                        self.cfg.timeout_s * 6)
        seen_urls, final, meta = set(), "", {}
        for line in out.splitlines():
            try:
                ev = json.loads(line)
            except json.JSONDecodeError:
                continue
            if ev.get("type") == "user":            # tool results: the only URLs a source may come from
                seen_urls |= set(URL_RE.findall(json.dumps(ev.get("message", {}), ensure_ascii=False)))
                if isinstance(ev.get("tool_use_result"), (dict, list, str)):
                    seen_urls |= set(URL_RE.findall(json.dumps(ev["tool_use_result"], ensure_ascii=False)))
            elif ev.get("type") == "result":
                meta, final = ev, ev.get("result") or ""
        if meta.get("is_error"):
            raise ClaudeCodeError(f"claude: {meta.get('result') or meta.get('terminal_reason')}"[:300])
        return self.verified(final, seen_urls), self._usage(meta, self.name)

    @staticmethod
    def verified(text: str, seen_urls: set[str]) -> dict:
        """Split the brief from its source list; keep sources whose URL a tool returned; renumber them; drop
        markers that point at anything else."""
        from equitydesk.policy.provenance import domain, normalize_url
        body, _, tail = text.partition("Sources:")
        listed = SOURCE_LINE.findall(tail or text)
        seen = {normalize_url(u).rstrip("/.") for u in seen_urls}
        keep, remap = [], {}
        for mid, title, url in listed:
            u = normalize_url(url.rstrip(".,;"))
            if u and u.rstrip("/.") in seen and mid not in remap:
                remap[mid] = f"S{len(keep) + 1}"
                keep.append({"id": remap[mid], "url": u, "title": title.strip() or domain(u), "domain": domain(u)})
        body = re.sub(r"\[(S\d+)\]", lambda m: f"[{remap[m.group(1)]}]" if m.group(1) in remap else "",
                      body if tail else SOURCE_LINE.sub("", text))
        return {"text": body.strip(), "sources": keep}
