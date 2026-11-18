"""Create a project-local environment file without displaying the API key."""

from getpass import getpass
from pathlib import Path

from dotenv import set_key

BASE_DIR = Path(__file__).resolve().parent
ENV_PATH = BASE_DIR / ".env"
DEFAULT_MODEL = "claude-haiku-4-5-20251001"


def write_project_env(
    api_key: str, model: str, path: Path | None = None
) -> None:
    if not api_key.strip():
        raise ValueError("The API key must not be empty")
    if not model.strip():
        raise ValueError("The model must not be empty")
    if path is None:
        path = ENV_PATH
    set_key(path, "APP_ANTHROPIC_API_KEY", api_key)
    set_key(path, "APP_ANTHROPIC_MODEL", model)


def main() -> None:
    api_key = getpass("Anthropic API key (input hidden): ")
    model = input(f"Anthropic model [{DEFAULT_MODEL}]: ").strip() or DEFAULT_MODEL
    write_project_env(api_key, model)
    print(f"Project settings saved to {ENV_PATH.name}.")


if __name__ == "__main__":
    main()
