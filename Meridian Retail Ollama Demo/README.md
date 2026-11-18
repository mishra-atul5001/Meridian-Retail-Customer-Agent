# Meridian Retail Support-Draft Review Demo (Ollama)

This command-line classroom demo prepares a support reply for human review. It uses Ollama's local API and does not require an Anthropic account or API key. It does not send messages, approve refunds, or change case records.

## Setup

Install and start Ollama, then download the default model:

```powershell
ollama pull llama3.2
```

In this folder, install the Python dependencies:

```powershell
python -m pip install -r requirements.txt
```

The app connects to `http://localhost:11434` and uses the `llama3.2` model by default. To choose another model, run `python setup_env.py` and enter a model already installed in Ollama, or set `APP_OLLAMA_MODEL` in a project-local `.env`. No API key is requested, stored, or sent.

## Run

```powershell
python app.py --case C1
python app.py --case C2 --offline-demo
python app.py --case C3 --simulate-timeout
```

Run one command per case, substituting `C1`, `C2`, or `C3`. `--offline-demo` displays a labelled prerecorded synthetic result and makes no Ollama call. `--simulate-timeout` makes no Ollama call, clears the draft, and displays a manual fallback. Ollama/provider errors and invalid model output use the same safe unavailable result.

The app asks the model to return the required JSON shape and validates its fields, types, status, and policy references. Correct structure does not prove correct content: a human must review every draft before use.

## Tests

Tests use fake model functions and do not call Ollama:

```powershell
python -m pytest
```
