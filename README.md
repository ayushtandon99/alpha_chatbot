# AI Policy Assistant – Hackathon Starter Kit

This project helps you build a chatbot that can answer questions about your internal SOPs and policy documents using free and open-source tools (no API key required).

---
## Project Overview

This starter kit supports PART A development path:

- **Path A (Used)**: 100% free & open-source – No OpenAI or API key needed.
- **Path B** *(Optional)*: Uses OpenAI GPT models – requires an API key (not included here).

---

## Folder Structure

```bash
TEAM_ALPHA_CHATBOT/
│
├── docs/                # Place your SOP/Policy text files here (.pdf, .txt, .docx)
├── faiss_index/         # Stores generated FAISS vector index (auto-generated)
│
└── src/                 # All core source code
    ├── ingest.py            # Script to process documents and build FAISS index
    ├── chat_local.py        # CLI-based chatbot interface (terminal)
    ├── appnew.py            # Streamlit web app for chatbot interface
    ├── faiss_index/         # (Optional) FAISS file substructure
    └── temp/                # Temporary files or cache
```

---


##  Setup Instructions

###  Path A: Free/Open-Source (Flan-T5 + FAISS)

1. **Install dependencies**:
   ```bash
   pip install -r requirements_free.txt
   ```

2. **Ingest documents**:
   ```bash
   python src/ingest.py
   ```

3. **Option 1: Run CLI chatbot**:
   ```bash
   python src/chat_local.py
   ```

4. **Option 2: Run Streamlit web app**:
   ```bash
   streamlit run src/appnew.py
   ```

---

## Features

-  No API keys required
-  Uses FAISS for fast document search
-  Uses Hugging Face `google/flan-t5-base` for LLM
-  Multilingual support (English, Spanish, French, German)
-  CLI and Web UI chatbot option as STRAMLIT 

---
## Notes
- Place all your `.pdf`, `.docx`, or `.txt` files in the `docs/` folder before running `ingest.py`.
- FAISS index will be saved automatically inside `faiss_index/`.
---
## Authors & Credits
Team ALPHA -
1.Pavan Teli
2.Harshal Thorat
3.Ayush Tondon
4.Anshuman Rajak

Built for the AI Policy Assistant Hackathon 💡  
Powered by LangChain, HuggingFace, Streamlit, and FAISS.

