---
title: PolicyPulse AI Sandbox
emoji: 🛡️
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

# PolicyPulse AI | Content Moderation Sandbox

A high-fidelity OpenEnv for benchmarking automated moderation policies with fairness constraints. Developed for the Meta-PyTorch Hackathon.

## 👨‍⚖️ Evaluation Guide for Hackathon Judges

This project features a **dual-use architecture** to satisfy strict automated baseline graders while giving human judges rich visual capabilities.

### 1. Automated Baseline Testing (Strict Compliance)
The environment complies strictly with the OpenEnv criteria. You can run the autonomous scripts without modification. The `inference.py` script automatically uses secure environment variables:
```bash
export API_BASE_URL="https://api.openai.com/v1"
export MODEL_NAME="gpt-4o-mini"
export HF_TOKEN="your-api-key"
python inference.py clear_cut_moderation 42
```
It outputs the strict `[START]`, `[STEP]`, and `[END]` logging required for baseline reproducibility.

### 2. Interactive Sandbox & APIs (Dynamic Overrides)
If you want to manually test the API with your API keys without fighting server-level configs, our FastAPI deployment allows live parameter injection:
* **Web UI**: Navigate to the Space URL root to access the interactive Operation Center. You can enter your custom `Base URL` and `API Key` directly into the UI to push moderation tasks to your preferred LLM instantly.
* **REST API**: Open the `/docs` route. The payload for both `/evaluate` and `/predict_and_step` accepts optional overrides (`api_base_url`, `model_name`, `api_key`). If left blank, they gracefully fall back to the Hugging Face Space environment settings.

## 🚀 Deployment Instructions

This project is designed to run on Hugging Face Spaces using Docker.

### Local Development
```bash
pip install -r requirements.txt
python3 app.py
```

### Grader Compliance
The environment uses `SocialStreamModerationEnv` and is compliant with Task 1, 2, and 3 requirements.

### License
MIT
