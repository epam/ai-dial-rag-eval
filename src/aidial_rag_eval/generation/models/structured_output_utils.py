from typing import Literal

StructuredOutputMethod = Literal["function_calling", "json_schema"]


def get_example_output_note(method: StructuredOutputMethod) -> str:
    if method == "function_calling":
        return (
            "\nNote: the example output above illustrates the expected data structure. "
            "When using function calling, return the data via a tool call with the same structure."
        )
    return ""


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
