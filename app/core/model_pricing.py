from typing import TypedDict


class ModelPricing(TypedDict):
    input: float
    output: float


# Prices are in USD per 1 million tokens.
# Add verified prices for each model before calculating costs.
MODEL_PRICING: dict[str, ModelPricing] = {
    # Example structure:
    # "provider/model-id": {
    #     "input": 0.0,
    #     "output": 0.0,
    # },
}