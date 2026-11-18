"""CLI for preparing Meridian Retail support drafts for human review with Ollama."""

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Callable
from urllib.request import Request, urlopen

from dotenv import dotenv_values

BASE_DIR = Path(__file__).resolve().parent
OLLAMA_CHAT_URL = "http://localhost:11434/api/chat"
DEFAULT_MODEL = "llama3.2"
RESPONSE_KEYS = {
    "draft_reply",
    "evidence_refs",
    "missing_information",
    "review_status",
}
REVIEW_STATUSES = {
    "READY_FOR_HUMAN_REVIEW",
    "NEEDS_INFORMATION",
    "BLOCKED",
}
FALLBACK_MESSAGE = (
    "A draft could not be prepared. Please review the supplied facts and policy "
    "manually before responding."
)
ModelFunction = Callable[[dict, str, str], str]
RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "draft_reply": {"type": "string"},
        "evidence_refs": {"type": "array", "items": {"type": "string"}},
        "missing_information": {"type": "array", "items": {"type": "string"}},
        "review_status": {
            "type": "string",
            "enum": sorted(REVIEW_STATUSES),
        },
    },
    "required": sorted(RESPONSE_KEYS),
    "additionalProperties": False,
}


def load_case(case_id: str) -> dict:
    with (BASE_DIR / "cases.json").open(encoding="utf-8") as cases_file:
        cases = json.load(cases_file)
    if case_id not in cases:
        raise ValueError(f"Unknown case: {case_id}")
    return cases[case_id]


def load_policy() -> str:
    return (BASE_DIR / "policy.md").read_text(encoding="utf-8")


def policy_ids(policy: str) -> set[str]:
    return set(re.findall(r"(?m)^(P\d+)\.", policy))


def validate_result(raw_result: str, policy: str) -> dict:
    result = json.loads(raw_result)
    if type(result) is not dict or set(result) != RESPONSE_KEYS:
        raise ValueError("Model response does not have the required JSON fields")
    if not isinstance(result["draft_reply"], str) or not result["draft_reply"].strip():
        raise ValueError("draft_reply must be non-empty text")
    for field in ("evidence_refs", "missing_information"):
        value = result[field]
        if type(value) is not list or any(
            not isinstance(item, str) or not item.strip() for item in value
        ):
            raise ValueError(f"{field} must be a list of non-empty strings")
    if not set(result["evidence_refs"]).issubset(policy_ids(policy)):
        raise ValueError("evidence_refs must contain policy IDs from the supplied policy")
    if (
        not isinstance(result["review_status"], str)
        or result["review_status"] not in REVIEW_STATUSES
    ):
        raise ValueError("review_status is not an allowed status")
    return result


def case_guidance(case: dict) -> str:
    if case.get("delivery_date") is None:
        return (
            "The delivery_date field is missing. Ask the customer for it and do "
            "not say the two-day review rule can be applied yet. Set "
            'missing_information to ["delivery_date"], review_status to '
            "NEEDS_INFORMATION, and cite P2 and P4. Suggested customer-facing "
            'wording: "What was the delivery date? I need it to apply the '
            'two-day rule."'
        )
    question = case.get("customer_question", "").lower()
    if case.get("issue_status") == "open" and "resolved" in question:
        return (
            "The customer asks whether the issue is resolved, but its status is "
            "open and the request is still under review. Say you cannot confirm "
            "resolution; do not call the issue blocked, state it is resolved, or "
            "promise follow-up. Set missing_information to [], review_status to "
            "BLOCKED, and cite P3 and P4. Use BLOCKED only as internal metadata. "
            'Suggested customer-facing wording: "The replacement request is '
            'still under review, so I cannot confirm that the issue is resolved."'
        )
    if (
        "yesterday" in question
        and "broken" in case.get("issue_details", "").lower()
        and any(word in question for word in ("return", "refund"))
    ):
        return (
            "The customer says the damaged item arrived yesterday and "
            "delivery_date is supplied. Do not ask for the delivery date or list "
            "it as missing. It may be referred for human return review; do not "
            "imply refund approval. Set missing_information to [], "
            "review_status to READY_FOR_HUMAN_REVIEW, and cite P1 and P4. "
            'Suggested customer-facing wording: "The reported damage may be '
            'referred for human return review. This does not mean a refund has '
            'been approved."'
        )
    return (
        "Use the supplied policy to identify any missing required information "
        "and choose the appropriate review_status."
    )


def request_model(case: dict, policy: str, model: str) -> str:
    prompt = (
        "Write a concise customer-service reply using only the supplied case "
        "facts and policy. Return only one JSON object matching the required "
        "schema. draft_reply must be a non-empty sentence. Do not use "
        "placeholders, invent actions or follow-up promises, approve refunds, "
        "or claim an open issue is resolved. This is for human review only; "
        "never send a message or change a record.\n\n"
        f"CASE-SPECIFIC GUIDANCE:\n{case_guidance(case)}\n\n"
        f"SUPPLIED FACTS:\n{json.dumps(case, ensure_ascii=False)}\n\n"
        f"SUPPLIED POLICY:\n{policy}"
    )
    request_body = json.dumps(
        {
            "model": model,
            "stream": False,
            "format": RESPONSE_SCHEMA,
            "messages": [{"role": "user", "content": prompt}],
        }
    ).encode("utf-8")
    request = Request(
        OLLAMA_CHAT_URL,
        data=request_body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=20) as response:
        response_data = json.loads(response.read().decode("utf-8"))
    content = response_data["message"]["content"]
    if not isinstance(content, str):
        raise ValueError("Ollama response content was not text")
    return content


def unavailable_result() -> dict:
    return {
        "draft_reply": "",
        "evidence_refs": [],
        "missing_information": [],
        "review_status": "UNAVAILABLE",
    }


def get_model_result(
    case: dict,
    policy: str,
    model: str,
    model_function: ModelFunction | None = None,
) -> dict:
    if model_function is None:
        raw_result = request_model(case, policy, model)
    else:
        raw_result = model_function(case, policy, model)
    return validate_result(raw_result, policy)


def load_offline_result(case_id: str, policy: str) -> tuple[dict, str]:
    with (BASE_DIR / "demo_outputs.json").open(encoding="utf-8") as demo_file:
        demo = json.load(demo_file)
    label = demo["label"]
    result = validate_result(json.dumps(demo["results"][case_id]), policy)
    if not isinstance(label, str) or not label.strip():
        raise ValueError("Offline demo must have a descriptive label")
    return result, label


def print_sections(case: dict, policy: str, result: dict, note: str | None = None) -> None:
    print("SUPPLIED FACTS")
    print(json.dumps(case, ensure_ascii=False, indent=2))
    print("\nSUPPLIED POLICY")
    print(policy.rstrip())
    print("\nDRAFT RESULT FOR HUMAN REVIEW")
    if note:
        print(note)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if result["review_status"] == "UNAVAILABLE":
        print(f"Manual fallback: {FALLBACK_MESSAGE}")


def main(argv: list[str] | None = None, model_function: ModelFunction | None = None) -> int:
    parser = argparse.ArgumentParser(description="Draft a Meridian Retail support reply.")
    parser.add_argument("--case", required=True, choices=("C1", "C2", "C3"))
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--simulate-timeout", action="store_true")
    modes.add_argument("--offline-demo", action="store_true")
    args = parser.parse_args(argv)

    project_settings = dotenv_values(BASE_DIR / ".env", interpolate=False)
    case = load_case(args.case)
    policy = load_policy()
    note = None

    if args.simulate_timeout:
        result = unavailable_result()
        note = "Simulated timeout: no Ollama API was called."
    elif args.offline_demo:
        try:
            result, label = load_offline_result(args.case, policy)
            note = f"{label}\nNo Ollama API was called."
        except Exception:
            result = unavailable_result()
            note = "Offline demo data was unavailable or invalid. No Ollama API was called."
    else:
        model = os.environ.get("APP_OLLAMA_MODEL")
        if model is None:
            model = project_settings.get("APP_OLLAMA_MODEL")
        model = model or DEFAULT_MODEL
        try:
            result = get_model_result(
                case, policy, model, model_function=model_function
            )
        except Exception:
            result = unavailable_result()
            note = (
                "Ollama or its model response was unavailable or invalid. "
                "Check that Ollama is running and the model is installed."
            )

    print_sections(case, policy, result, note)
    return 0


if __name__ == "__main__":
    sys.exit(main())
