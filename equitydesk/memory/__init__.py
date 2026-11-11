"""Memory & state.

Short-term (within a run):  LangGraph state, checkpointed after every node to data/checkpoints.db
                             (SqliteSaver), so a crashed or budget-stopped run resumes where it stopped,
                             and each report's follow-up Q&A keeps its own conversation thread.
Long-term (across runs):     memory/store.py, a SQLite store of past calls (with the price at the time),
                             the user's notes per ticker, the user's risk profile, and the run ledger.
Reflection:                  memory/track.py scores every past call against what the stock then did versus
                             the benchmark; the PM sees its own track record on a ticker before it decides.
"""
