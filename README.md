# ResumeLLME — AI Resume Generator

ResumeLLME is a Streamlit app that transforms an uploaded CV into a polished resume using LLMs and LaTeX templates.

## Features

- LLM-powered resume generation from CV text
- Optional resume improvement flow using LLM prompts
- Multi-provider support (OpenAI and Gemini)
- Quota/rate-limit handling with controlled failures
- Automatic fallback mode when API or quota fails
- Response caching to reduce repeated API calls
- PDF, LaTeX, and JSON downloads

## Tech Stack

- Python 3.12
- Streamlit
- OpenAI Python SDK
- Google Generative AI SDK (Gemini)
- Jinja2 (templating)
- LaTeX toolchain for PDF rendering

## Installation

### 1) Clone the repository

```bash
git clone https://github.com/Karunesh17/ResumeLLME.git
cd ResumeLLME
```

### 2) Create and activate a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3) Install Python dependencies

```bash
pip install -r requirements.txt
```

### 4) Install LaTeX packages (Linux)

```bash
xargs sudo apt install -y < packages.txt
```

### 5) Configure environment variables

Copy `.env.example` to `.env` and set your keys.

```bash
cp .env.example .env
```

## Usage

Run the app:

```bash
streamlit run src/Main.py
```

If `streamlit` is not on PATH, run:

```bash
venv/bin/streamlit run src/Main.py
```

## Screenshots

- App Home: _Add screenshot here_
- Resume Generation Flow: _Add screenshot here_
- Fallback/Quota Message UI: _Add screenshot here_

## Notes

- No API keys are hardcoded in the source.
- Keep secrets in `.env` or your deployment secret manager.
