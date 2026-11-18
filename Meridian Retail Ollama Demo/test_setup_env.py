from dotenv import dotenv_values

import setup_env


def test_setup_writes_model_without_requesting_or_printing_a_key(
    tmp_path, monkeypatch, capsys
):
    env_path = tmp_path / ".env"
    monkeypatch.setattr(setup_env, "ENV_PATH", env_path)
    monkeypatch.setattr("builtins.input", lambda _prompt: "")

    setup_env.main()

    output = capsys.readouterr().out
    values = dotenv_values(env_path)
    assert values == {"APP_OLLAMA_MODEL": setup_env.DEFAULT_MODEL}
    assert "API key" not in output
    assert "saved" in output


def test_setup_writes_to_project_only_path(tmp_path):
    env_path = tmp_path / ".env"
    setup_env.write_project_env("qwen2.5:7b", env_path)
    assert dotenv_values(env_path) == {"APP_OLLAMA_MODEL": "qwen2.5:7b"}
