from typing import Literal

StructuredOutputMethod = Literal["function_calling", "json_schema"]


def get_structured_output_instruction(method: StructuredOutputMethod) -> str:
    if method == "function_calling":
        return (
            "\nIMPORTANT: Complete this entire task in a SINGLE response. "
            "Call the tool EXACTLY ONCE with ALL results in that one call."
        )
    return (
        "\nIMPORTANT: Complete this entire task in a SINGLE response. "
        "Respond with a SINGLE JSON object containing ALL results."
    )
