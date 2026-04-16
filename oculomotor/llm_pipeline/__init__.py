"""LLM simulation pipeline — natural-language → SimulationScenario → Figure.

Submodules
----------
scenario   Pydantic schema (SimulationScenario, SimulationComparison, Patient)
runner     Stimulus builder + simulator wiring + matplotlib figure generator
simulate   CLI entry point + Claude API call (call_llm / main)
"""
