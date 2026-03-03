import re


def clean_json_string(raw_output):
    """
    Removes code fences and whitespace from a model output, leaving only valid JSON.
    """
    # Remove ```json ... ``` or ``` ... ```
    cleaned = re.sub(r"```(?:json)?\s*([\s\S]*?)```", r"\1", raw_output, flags=re.IGNORECASE)
    return cleaned.strip()