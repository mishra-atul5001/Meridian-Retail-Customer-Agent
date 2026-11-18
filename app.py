"""CLI for preparing Meridian Retail support drafts for human review."""

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Callable

from dotenv import dotenv_values

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL = "claude-haiku-4-5-20251001"
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


def request_model(case: dict, policy: str, model: str, api_key: str) -> str:
    from anthropic import Anthropic

    client = Anthropic(api_key=api_key, timeout=20.0, max_retries=0)
    prompt = (
        "Prepare a support draft using only the supplied case facts and policy. "
        "Do not invent facts or claim an action was taken. Do not say an issue is "
        "resolved when its status is open. Do not imply a refund is approved. "
        "Return only a JSON object with exactly these fields: draft_reply (text), "
        "evidence_refs (list of policy IDs), missing_information (list of field "
        "names), and review_status (one of READY_FOR_HUMAN_REVIEW, "
        "NEEDS_INFORMATION, BLOCKED). Use NEEDS_INFORMATION when required "
        "information is missing and BLOCKED when the request cannot be supported "
        "by the policy. This is for human review only; never send a message, "
        "approve a refund, or change a record.\n\n"
        f"SUPPLIED FACTS:\n{json.dumps(case, ensure_ascii=False)}\n\n"
        f"SUPPLIED POLICY:\n{policy}"
    )
    response = client.messages.create(
        model=model,
        max_tokens=500,
        messages=[{"role": "user", "content": prompt}],
    )
    text_blocks = [
        block.text for block in response.content if getattr(block, "type", None) == "text"
    ]
    if len(text_blocks) != 1:
        raise ValueError("Model response did not contain exactly one text block")
    return text_blocks[0]


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
    api_key: str,
    model: str,
    model_function: ModelFunction | None = None,
) -> dict:
    if model_function is None:
        raw_result = request_model(case, policy, model, api_key)
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
        note = "Simulated timeout: no model API was called."
    elif args.offline_demo:
        try:
            result, label = load_offline_result(args.case, policy)
            note = f"{label}\nNo model API was called."
        except Exception:
            result = unavailable_result()
            note = "Offline demo data was unavailable or invalid. No model API was called."
    else:
        api_key = os.environ.get("APP_ANTHROPIC_API_KEY")
        if api_key is None:
            api_key = project_settings.get("APP_ANTHROPIC_API_KEY")
        model = os.environ.get("APP_ANTHROPIC_MODEL")
        if model is None:
            model = project_settings.get("APP_ANTHROPIC_MODEL")
        model = model or DEFAULT_MODEL
        if not api_key:
            result = unavailable_result()
            note = "APP_ANTHROPIC_API_KEY is not configured."
        else:
            try:
                result = get_model_result(
                    case, policy, api_key, model, model_function=model_function
                )
            except Exception:
                result = unavailable_result()
                note = "The provider or model response was unavailable or invalid."

    print_sections(case, policy, result, note)
    return 0


if __name__ == "__main__":
    sys.exit(main())
