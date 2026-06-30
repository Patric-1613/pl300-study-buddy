import os
import re
import json
import time
from pathlib import Path

from liteparse import LiteParse
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.tools.retriever import create_retriever_tool
from langchain.tools import tool
from langchain.agents import create_agent

NOTES_PATH = "pl300_notes.pdf"
WEAK_TOPICS_FILE = Path("weak_topics.json")
DB_PATH = "./chroma_db_pl300"


def build_agent():
    # --- Parse ---
    parser = LiteParse(output_format="text", ocr_enabled=False)
    parsed = parser.parse(NOTES_PATH)

    # --- Chunk ---
    primary_chunks = re.split(r'(?=Question: \d+)', parsed.text)
    primary_chunks = [c.strip() for c in primary_chunks if c.strip()]

    MAX_CHARS = 1200
    fallback = RecursiveCharacterTextSplitter(chunk_size=MAX_CHARS, chunk_overlap=100)
    chunks = []
    for c in primary_chunks:
        if len(c) <= MAX_CHARS:
            chunks.append(c)
        else:
            match = re.match(r'(Question:\s*\d+)', c)
            header = match.group(1) if match else "Question: unknown"
            for i, sc in enumerate(fallback.split_text(c)):
                chunks.append(sc if i == 0 else f"[{header}, continued]\n{sc}")

    documents = [Document(page_content=c) for c in chunks]

    # --- Embed + Store ---
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = Chroma(
        embedding_function=embeddings,
        collection_name="pl300_notes",
        persist_directory=DB_PATH,
    )

    if not os.path.exists(DB_PATH) or not os.listdir(DB_PATH):
        BATCH_SIZE = 40
        for i in range(0, len(documents), BATCH_SIZE):
            batch = documents[i:i + BATCH_SIZE]
            vectorstore.add_documents(batch)
            print(f"Embedded chunks {i} to {i + len(batch) - 1}")
            time.sleep(15)
    else:
        print("Vector DB loaded from disk.")

    # --- Retriever Tool ---
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5}) # it was 3 initally, i am trying 5 now. cat
    retriever_tool = create_retriever_tool(
        retriever,
        name="search_pl300_notes",
        description="Search PL-300 study notes for Power BI topics.",
    )

    # --- Weak Topic Tools ---
    @tool
    def log_weak_topic(topic: str, note: str = "") -> str:
        """Record a PL-300 topic the user is struggling with."""
        data = []
        if WEAK_TOPICS_FILE.exists():
            data = json.loads(WEAK_TOPICS_FILE.read_text())
        data.append({"topic": topic, "note": note})
        WEAK_TOPICS_FILE.write_text(json.dumps(data, indent=2))
        return f"Logged '{topic}'. Total: {len(data)}."

    @tool
    def list_weak_topics() -> str:
        """List all topics the user has flagged as weak."""
        if not WEAK_TOPICS_FILE.exists():
            return "No weak topics logged yet."
        data = json.loads(WEAK_TOPICS_FILE.read_text())
        return "\n".join(f"- {d['topic']}: {d['note']}" for d in data)

    # --- Agent ---
# --- Agent ---
    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(
        model="gpt-5.4-mini",
        temperature=0.2,
        top_p=0.9,
        frequency_penalty=0.5,
        presence_penalty=0.1,
    )

    agent = create_agent(
        model=llm,
        tools=[retriever_tool, log_weak_topic, list_weak_topics],
        system_prompt=(
            "You are a PL-300 exam study buddy. Your knowledge comes ONLY from the "
            "user's study notes — always call search_pl300_notes before answering any "
            "factual question. Never answer from general knowledge alone. If the notes "
            "don't contain enough information, say so explicitly rather than guessing. "
            "Do not hallucinate exam questions or claim something is on the exam unless "
            "it appears in the retrieved notes. If the user expresses confusion or says "
            "they got something wrong, call log_weak_topic. Only log weak topics if the "
            "user explicitly says they are confused, struggling, or got something wrong. "
            "Do not log topics just because the user asked about them. "
            "Use list_weak_topics when asked to review progress."
        ),
    )
    return agent


def get_weak_topics():
    if not WEAK_TOPICS_FILE.exists():
        return []
    return json.loads(WEAK_TOPICS_FILE.read_text())