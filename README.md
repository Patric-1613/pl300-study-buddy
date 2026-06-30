# PL-300 Study Buddy

A RAG-powered AI agent that answers questions from your Power BI PL-300 exam question bank, tracks topics you struggle with, and serves a clean chat UI in the browser.

Built during the Virtusa 12-week Agentic AI internship programme.

## What it does

- Parses your PL-300 PDF question bank using liteparse
- Chunks content by question boundary and stores it in Chroma
- Uses OpenAI embeddings (text-embedding-3-small) for semantic search
- Runs a LangChain agent (gpt-5.4-mini) with three tools:
  - search_pl300_notes — retrieves relevant chunks from your notes
  - log_weak_topic — saves topics you struggle with to weak_topics.json
  - list_weak_topics — lists everything you have flagged so far
- Serves a Flask web UI at http://localhost:5000

## Setup

### 1. Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate

### 2. Install dependencies
pip install -r requirements.txt

### 3. Set up environment variables
cp .env.example .env
# Open .env and fill in your OpenAI API key

### 4. Add your PDF
Place your PL-300 question bank PDF in the project root named pl300_notes.pdf

### 5. Run the web app
python app.py
# Open http://localhost:5000

### 6. Or run the terminal version
python study_buddy.py

## Tech stack
- PDF parsing: liteparse
- Embeddings: OpenAI text-embedding-3-small
- Vector DB: Chroma
- LLM: gpt-5.4-mini
- Agent: LangChain create_agent
- Web: Flask + marked.js

## Author
Pratik Mugade — MSc Data Science, Kingston University
Built as part of the Virtusa GenAI Internship 2026
