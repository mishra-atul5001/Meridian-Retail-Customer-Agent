"""Set the model for this project without requiring or storing an API key."""

from pathlib import Path

from dotenv import set_key

BASE_DIR = Path(__file__).resolve().parent
ENV_PATH = BASE_DIR / ".env"
DEFAULT_MODEL = "llama3.2"


def write_project_env(model: str, path: Path | None = None) -> None:
    if not model.strip():
        raise ValueError("The model must not be empty")
    if path is None:
        path = ENV_PATH
    set_key(path, "APP_OLLAMA_MODEL", model)


def main() -> None:
    model = input(f"Ollama model [{DEFAULT_MODEL}]: ").strip() or DEFAULT_MODEL
    write_project_env(model)
    print(f"Project model setting saved to {ENV_PATH.name}.")


if __name__ == "__main__":
    main()
