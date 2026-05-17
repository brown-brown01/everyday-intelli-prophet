"""Prophet Arena hackathon bot.

A thin client that competes in the Prophet Arena prediction-market
benchmark. The decision brain combines an LLM forecaster with the V1
Bayesian belief engine (prophet_arena.bayesian_core):

    market quote ──► forecaster ──► bayes.temper ──► strategy.decide ──► intent

Entry points:
    python -m prophet_arena.harness   # offline scoring, no PA key needed
    python -m prophet_arena.bot       # live tick loop, needs PA_SERVER_API_KEY
"""
