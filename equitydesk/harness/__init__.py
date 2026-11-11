"""The harness: every model call goes through harness.llm.LLM, which meters, prices, bounds, retries,
repairs schema failures and traces it. Nothing else in the code base talks to a model SDK."""
