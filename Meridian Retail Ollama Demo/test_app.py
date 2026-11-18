import json
from io import BytesIO
from pathlib import Path

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
        app.load_case("C1"), app.load_policy(), "test-model", fake_model
    )
    assert result["review_status"] == "READY_FOR_HUMAN_REVIEW"
    assert called == [("C1", "test-model")]


def test_invalid_model_output_is_rejected():
    def fake_model(_case, _policy, _model):
        return '{"draft_reply": "incomplete"}'

    try:
        app.get_model_result(
            app.load_case("C1"), app.load_policy(), "test-model", fake_model
        )
    except ValueError:
        pass
    else:
        raise AssertionError("Invalid model structure was accepted")


def test_simulated_timeout_clears_draft_and_makes_no_call(capsys):
    def forbidden_model(*_args):
        raise AssertionError("simulate-timeout must not call Ollama")

    assert app.main(
        ["--case", "C1", "--simulate-timeout"], model_function=forbidden_model
    ) == 0
    output = capsys.readouterr().out
    assert "Simulated timeout: no Ollama API was called." in output
    assert "Manual fallback:" in output
    assert '"draft_reply": ""' in output
    assert '"review_status": "UNAVAILABLE"' in output


def test_offline_demo_is_labelled_and_makes_no_call(capsys):
    def forbidden_model(*_args):
        raise AssertionError("offline-demo must not call Ollama")

    assert app.main(["--case", "C2", "--offline-demo"], model_function=forbidden_model) == 0
    output = capsys.readouterr().out
    assert "PRERECORDED SYNTHETIC CLASSROOM RESULT" in output
    assert "No Ollama API was called." in output
    assert '"review_status": "NEEDS_INFORMATION"' in output


def test_ollama_request_uses_local_chat_api_and_json_schema(monkeypatch):
    seen = {}
    model_content = json.dumps(
        {
            "draft_reply": "A human reviewer can assess the request.",
            "evidence_refs": ["P1"],
            "missing_information": [],
            "review_status": "READY_FOR_HUMAN_REVIEW",
        }
    )

    def fake_urlopen(request, timeout):
        seen["url"] = request.full_url
        seen["timeout"] = timeout
        seen["body"] = json.loads(request.data.decode("utf-8"))
        return BytesIO(
            json.dumps({"message": {"content": model_content}}).encode("utf-8")
        )

    monkeypatch.setattr(app, "urlopen", fake_urlopen)
    result = app.get_model_result(app.load_case("C1"), app.load_policy(), "test-model")
    assert result["review_status"] == "READY_FOR_HUMAN_REVIEW"
    assert seen["url"] == "http://localhost:11434/api/chat"
    assert seen["timeout"] == 20
    assert seen["body"]["model"] == "test-model"
    assert seen["body"]["format"] == app.RESPONSE_SCHEMA
    assert seen["body"]["stream"] is False
    assert "delivery_date is supplied" in seen["body"]["messages"][0]["content"]


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


def test_bad_model_response_uses_unavailable_state(capsys):
    def invalid_model(*_args):
        return '{"draft_reply": "not the required structure"}'

    assert app.main(["--case", "C1"], model_function=invalid_model) == 0
    output = capsys.readouterr().out
    assert '"draft_reply": ""' in output
    assert '"review_status": "UNAVAILABLE"' in output
    assert "Manual fallback:" in output
    assert "Traceback" not in output


def test_provider_failure_uses_unavailable_state(monkeypatch, capsys):
    def failed_urlopen(*_args, **_kwargs):
        raise OSError("Ollama unavailable")

    monkeypatch.setattr(app, "urlopen", failed_urlopen)
    assert app.main(["--case", "C1"]) == 0
    output = capsys.readouterr().out
    assert '"draft_reply": ""' in output
    assert '"review_status": "UNAVAILABLE"' in output
    assert "Traceback" not in output
