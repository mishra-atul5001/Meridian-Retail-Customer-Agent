from dotenv import dotenv_values

import setup_env


def test_setup_writes_key_without_printing_it(tmp_path, monkeypatch, capsys):
    secret = "demo-secret-value"
    env_path = tmp_path / ".env"
    monkeypatch.setattr(setup_env, "ENV_PATH", env_path)
    monkeypatch.setattr(setup_env, "getpass", lambda _prompt: secret)
    monkeypatch.setattr("builtins.input", lambda _prompt: "")

    setup_env.main()

    output = capsys.readouterr().out
    values = dotenv_values(env_path)
    assert values["APP_ANTHROPIC_API_KEY"] == secret
    assert values["APP_ANTHROPIC_MODEL"] == setup_env.DEFAULT_MODEL
    assert secret not in output
    assert "saved" in output


def test_setup_writes_to_project_only_path(tmp_path):
    env_path = tmp_path / ".env"
    setup_env.write_project_env("another-secret", "test-model", env_path)
    assert env_path.is_file()
    assert dotenv_values(env_path) == {
        "APP_ANTHROPIC_API_KEY": "another-secret",
        "APP_ANTHROPIC_MODEL": "test-model",
    }
