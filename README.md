# Vibration analysis demo

A Streamlit app: upload a CSV of triaxial vibration data, get analysis plots and
a classification.

Access is gated by a shared password. The trained model is not in this
repository; it is fetched at startup from a private store.

## Configuration

All three are set as Streamlit Community Cloud secrets, or as environment
variables on a container host. Community Cloud exposes top-level secrets as
environment variables, so one name works for both.

| Name | Purpose |
| --- | --- |
| `APP_PASSWORD` | shared password for the page. Unset it and the gate disappears |
| `HF_ARTIFACT_REPO` | `owner/name` of the private Hugging Face dataset holding the model files |
| `HF_TOKEN` | Hugging Face token with read access to that dataset |

Without `HF_ARTIFACT_REPO` the app reads from a local `model_out/` directory
instead, which is how it runs during development.

## Deploying

Community Cloud: pick this repo and `app.py`, **set Python to 3.12** in advanced
settings, and paste the three secrets. Python 3.12 is required — scikit-learn
1.6.1 publishes no wheels beyond it, and the model pickle cannot be loaded
without that exact version.

## Running locally

```sh
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt
ART_DIR=/path/to/model_out .venv/bin/streamlit run app.py
```
