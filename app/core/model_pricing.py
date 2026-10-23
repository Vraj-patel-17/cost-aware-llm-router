from typing import TypedDict


class ModelPricing(TypedDict):
    input: float
    output: float


# Prices are in USD per 1 million tokens.
# Add verified prices for each model before calculating costs.
MODEL_PRICING: dict[str, ModelPricing] = {
    "gemini/gemini-3.5-flash-lite": {
        "input": 0.30,
        "output": 2.50,
    },
    "groq/openai/gpt-oss-120b": {
        "input": 0.15,
        "output": 0.60,
    },
}