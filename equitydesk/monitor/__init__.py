"""The market monitor: a background process that collects official releases, media reports, per-stock news,
retail sentiment and prediction markets, scores every item with a model, pushes the important ones and writes
short sourced reports on big events. Run it with `python -m equitydesk.monitor` (or start_monitor.bat).

    sources.py   what is collected (RSS, 华尔街见闻 7x24, Yahoo per ticker), each with a source tier
    signals.py   index quotes, Reddit mentions, StockTwits bull/bear, Polymarket and Kalshi odds
    triage.py    clustering of duplicate stories, ticker matching, model scoring, the AI overview
    reports.py   web-researched briefs on high-scoring events, with cited sources only
    notify.py    desktop toast and optional ntfy.sh phone push
    store.py     its own SQLite file (WAL), shared by the service and the UI
    service.py   the loop: control (run / pause / stop), heartbeat, daily cost ceiling
"""
