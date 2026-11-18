import json
from pathlib import Path
from types import SimpleNamespace

import anthropic
import app

PROJECT_DIR = Path(__file__).resolve().parent


def test_successful_c1_result_uses_fake_model():
    called = []

    def fake_model(case, policy, model):
        called.append((case["case_id"], model))
        return json.dumps(
            {
                "draft_reply": "A human reviewer can assess the return request.",
                "evidence_refs": ["P1", "P4"],
                "missing_information": [],
                "review_status": "READY_FOR_HUMAN_REVIEW",
            }
        )

    result = app.get_model_result(
        app.load_case("C1"), app.load_policy(), "test-key", "test-model", fake_model
    )
    assert result["review_status"] == "READY_FOR_HUMAN_REVIEW"
    assert called == [("C1", "test-model")]


def test_invalid_model_output_is_rejected():
    def fake_model(_case, _policy, _model):
        return '{"draft_reply": "incomplete"}'

    try:
        app.get_model_result(
            app.load_case("C1"), app.load_policy(), "test-key", "test-model", fake_model
        )
    except ValueError:
        pass
    else:
        raise AssertionError("Invalid model structure was accepted")


def test_simulated_timeout_clears_draft_and_makes_no_call(capsys):
    def forbidden_model(*_args):
        raise AssertionError("simulate-timeout must not call a model")

    assert app.main(
        ["--case", "C1", "--simulate-timeout"], model_function=forbidden_model
    ) == 0
    output = capsys.readouterr().out
    assert "Simulated timeout: no model API was called." in output
    assert "Manual fallback:" in output
    assert '"draft_reply": ""' in output
    assert '"review_status": "UNAVAILABLE"' in output


def test_offline_demo_is_labelled_and_makes_no_call(capsys):
    def forbidden_model(*_args):
        raise AssertionError("offline-demo must not call a model")

    assert app.main(["--case", "C2", "--offline-demo"], model_function=forbidden_model) == 0
    output = capsys.readouterr().out
    assert "PRERECORDED SYNTHETIC CLASSROOM RESULT" in output
    assert "No model API was called." in output
    assert '"review_status": "NEEDS_INFORMATION"' in output


def test_anthropic_client_uses_explicit_key_timeout_and_retries(monkeypatch):
    seen = {}
    model_json = json.dumps(
        {
            "draft_reply": "A human reviewer can assess the request.",
            "evidence_refs": ["P1"],
            "missing_information": [],
            "review_status": "READY_FOR_HUMAN_REVIEW",
        }
    )

    class FakeAnthropic:
        def __init__(self, **kwargs):
            seen["client"] = kwargs
            self.messages = self

        def create(self, **kwargs):
            seen["request"] = kwargs
            return SimpleNamespace(
                content=[SimpleNamespace(type="text", text=model_json)]
            )

    monkeypatch.setattr(anthropic, "Anthropic", FakeAnthropic)
    result = app.get_model_result(
        app.load_case("C1"), app.load_policy(), "application-key", "chosen-model"
    )
    assert result["review_status"] == "READY_FOR_HUMAN_REVIEW"
    assert seen["client"] == {
        "api_key": "application-key",
        "timeout": 20.0,
        "max_retries": 0,
    }
    assert seen["request"]["model"] == "chosen-model"


def test_all_required_input_and_output_files_exist():
    expected = {
        "policy.md",
        "cases.json",
        "app.py",
        "README.md",
        "requirements.txt",
        "setup_env.py",
        ".env.example",
        ".gitignore",
        "demo_outputs.json",
        "test_app.py",
        "test_setup_env.py",
    }
    assert expected.issubset({path.name for path in PROJECT_DIR.iterdir()})
    cases = json.loads((PROJECT_DIR / "cases.json").read_text(encoding="utf-8"))
    demo = json.loads((PROJECT_DIR / "demo_outputs.json").read_text(encoding="utf-8"))
    assert set(cases) == {"C1", "C2", "C3"}
    assert set(demo["results"]) == {"C1", "C2", "C3"}
    for case_id in cases:
        result, label = app.load_offline_result(case_id, app.load_policy())
        assert result["review_status"] in app.REVIEW_STATUSES
        assert label.startswith("PRERECORDED SYNTHETIC")


def test_bad_model_response_uses_unavailable_state(capsys, monkeypatch):
    monkeypatch.setenv("APP_ANTHROPIC_API_KEY", "test-key")

    def invalid_model(*_args):
        return '{"draft_reply": "not the required structure"}'

    assert app.main(["--case", "C1"], model_function=invalid_model) == 0
    output = capsys.readouterr().out
    assert '"draft_reply": ""' in output
    assert '"review_status": "UNAVAILABLE"' in output
    assert "Manual fallback:" in output
    assert "Traceback" not in output


def test_global_anthropic_key_is_not_used(capsys, monkeypatch):
    monkeypatch.delenv("APP_ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "must-not-be-used")
    assert app.main(["--case", "C1"]) == 0
    output = capsys.readouterr().out
    assert '"review_status": "UNAVAILABLE"' in output
    assert "must-not-be-used" not in output
