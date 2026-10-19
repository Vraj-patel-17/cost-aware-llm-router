from app.core.model_pricing import MODEL_PRICING

def calculate_cost(
    model: str,
    input_tokens: int,
    output_tokens: int,
) -> float | None:
    pricing = MODEL_PRICING.get(model)

    if pricing is None:
        return None

    input_cost = input_tokens * pricing["input"]
    output_cost = output_tokens * pricing["output"]

    return (input_cost + output_cost) / 1_000_000