
# 🧠 AI Policy Assistant Hackathon Starter Kit

This starter kit supports two development paths:
- **Path A**: Free and open-source (no API keys required)
- **Path B**: OpenAI-powered (requires API key, free tier supported)

## Folder Structure
- `docs/`: Place your SOP/Policy text files here
- `src/`: Source code for document ingestion and querying

## Setup Instructions

### Path A: Free/Open-Source
```bash
pip install -r requirements_free.txt
python src/ingest.py
python src/chat_local.py
```

### Path B: OpenAI-Powered
```bash
pip install -r requirements_openai.txt
# Add OPENAI_API_KEY to your environment
python src/ingest_openai.py
python src/chat_openai.py
```
