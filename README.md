# Meridian-Retail-Customer-Agent

This small command-line classroom demo prepares a support reply for human review. It does not send messages, approve refunds, or change case records.

## Setup

Use Python 3.10 or later. From this folder, install the pinned dependencies:

```text
python -m pip install -r requirements.txt
```

Run `python setup_env.py` to enter the Anthropic API key using hidden input and choose a model. The script writes a project-local `.env`; it does not display the key. Alternatively, copy `.env.example` to `.env` and fill in the values locally.

The application reads only `APP_ANTHROPIC_API_KEY` for the key and `APP_ANTHROPIC_MODEL` for the model. The default model is `claude-haiku-4-5-20251001`. Do not commit `.env`.

## Run

```text
python app.py --case C1
python app.py --case C2 --offline-demo
python app.py --case C3 --simulate-timeout
```

Each run prints supplied facts, supplied policy, and a draft result for human review. `--offline-demo` displays a labelled prerecorded synthetic result and makes no API call. `--simulate-timeout` makes no API call, clears the draft, and displays a manual fallback. Provider errors and invalid model output use the same safe unavailable result. The application validates the response shape and policy references; that validation does not establish that the draft's content is correct. A human must review it.

## Tests

Run the API-free test suite with:

```text
python -m pytest
```
