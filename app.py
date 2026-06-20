from __future__ import annotations

import os, time
from pathlib import Path

import requests
import streamlit as st

from langchain_core.prompts import ChatPromptTemplate
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

ROOT_DIR = Path(__file__).resolve().parent
DEFAULT_QDRANT_PATH = ROOT_DIR / ".qdrant_store"
DEFAULT_COLLECTION_NAME = "medical_passages"
DEFAULT_EMBED_MODEL = "nomic-ai/nomic-embed-text-v1.5"
DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_GENERATION_URL = os.environ.get(
"MED_AI_GENERATION_URL", "http://localhost:8988/generate/")

@st.cache_resource(show_spinner=False)
def load_embedder(model_pth: str):
    return SentenceTransformer(
        model_pth,
        trust_remote_code=True,
        device="cpu",
    )

@st.cache_resource(show_spinner=False)
def get_qdrant_client(qdrant_path: str):
    return QdrantClient(path=qdrant_path)

def retrieve_context(
    client,
    embedder,
    collection_name: str,
    query: str,
    top_k: int,
) -> list[dict[str, str]]:
    query_vector = embedder.encode(
        [query],
        normalize_embeddings=True,
    )[0]
    response = client.query_points(
        collection_name=collection_name,
        query=query_vector,
        limit=top_k,
        with_payload=True,
    )
    
    results: list[dict[str, str]] = []
    for hit in response.points:
        payload = hit.payload or {}
        results.append(
            {
                "id": str(payload.get("id", hit.id)),
                "text": str(payload.get("text", "")),
                "score": f"{float(hit.score):.4f}",
            }
        )
    print("result_ids:", list(map(lambda x: x["id"], results)))
    return results

def compute_retrieval_metrics(sources):

    scores = [
        float(s["score"])
        for s in sources
    ]

    if not scores:
        return {
            "avg": 0.0,
            "max": 0.0,
        }

    return {
        "avg": sum(scores) / len(scores),
        "max": max(scores),
    }

def build_prompt(question: str, contexts: list[dict[str, str]]) -> str:
    """Build the single-turn prompt using ChatPromptTemplate.

    This gemma3 model currently expects a pipe-style prompt with explicit markers; we use
    ChatPromptTemplate to format the central human message and then wrap it
    with the <start_of_turn>/<end_of_turn> markers required by the model.
    """
    context_block = "\n\n".join(
        f"{item['text']}"
        for _, item in enumerate(contexts)
    )

    if not context_block:
        context_block = "No retrieved context found."

    prompt_template = ChatPromptTemplate.from_messages(
        [("human", "Context:\n{context_block}\n\nQuestion:\n{question}")]
    )

    human_body = prompt_template.format(context_block=context_block, question=question)
    return f"<start_of_turn>user\n{human_body}<end_of_turn>\n<start_of_turn>model\n"

def stream_answer_from_endpoint(
    generation_url: str,
    prompt: str,
    max_length: int,
):
    response = requests.post(
        generation_url,
        json={
            "prompt": prompt,
            "max_length": max_length,
        },
        timeout=300,
    )
    response.raise_for_status()
    payload = response.json()
    print("RAW RESPONSE:", payload)
    answer = (
        payload.get("generated_text")
        or payload.get("response")
        or payload.get("text")
        or ""
    )
    
    streaming = answer.split()
    if not streaming:
        return
        
    lastword = streaming.pop()
    for word in streaming:
        yield word + " "
        time.sleep(0.2)
    yield lastword

def render_sources(sources: list[dict[str, str]]) -> None:
    if not sources:
        st.info("No supporting passages were retrieved for this question.")
        return

    for index, source in enumerate(sources, start=1):
        with st.expander(f"Source {index} | id={source['id']} | score={source['score']}", expanded=index == 1):
            st.write(source["text"])


st.set_page_config(page_title="Med Chatbot", page_icon="🩺", layout="wide")

st.markdown(
    """
    <style>
    .stApp {
        background: radial-gradient(circle at top left, rgba(14, 165, 233, 0.14), transparent 35%),
                    linear-gradient(180deg, #f6fbff 0%, #eef6f2 100%);
    }
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1200px;
    }
    .hero-card {
        background: rgba(255, 255, 255, 0.82);
        border: 1px solid rgba(15, 23, 42, 0.08);
        border-radius: 24px;
        padding: 1.25rem 1.5rem;
        box-shadow: 0 18px 60px rgba(15, 23, 42, 0.08);
        backdrop-filter: blur(8px);
    }
    .hero-title {
        font-size: 2rem;
        font-weight: 800;
        color: #0f172a;
        margin-bottom: 0.25rem;
    }
    .hero-subtitle {
        color: #475569;
        font-size: 1rem;
        line-height: 1.5;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero-card">
      <div class="hero-title">Medical RAG Chatbot</div>
      <div class="hero-subtitle">Local fine-tuned Gemma model + Qdrant retrieval over your parquet corpus.</div>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Settings")
    qdrant_path = st.text_input("Qdrant storage path", value=str(DEFAULT_QDRANT_PATH))
    collection_name = st.text_input("Collection name", value=DEFAULT_COLLECTION_NAME)
    ollama_base_url = st.text_input("Ollama base URL", value=DEFAULT_OLLAMA_BASE_URL)
    embed_model = st.text_input("Ollama embedding model", value=DEFAULT_EMBED_MODEL)
    generation_url = st.text_input("Generation endpoint", value=DEFAULT_GENERATION_URL)
    top_k = st.slider("Retrieved passages", min_value=1, max_value=4, value=2)
    max_length = st.slider("Answer length", min_value=128, max_value=2048, value=2000, step=64)
    clear_chat = st.button("Clear chat", use_container_width=True)

if clear_chat:
    st.session_state.messages = []
    st.rerun()

if "messages" not in st.session_state:
    st.session_state.messages = []

embedder = load_embedder(embed_model)
client = get_qdrant_client(qdrant_path)

st.caption(f"Generation endpoint: {generation_url}. Around 40k+ corpus texts in Qdrant vectorDB.")
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
st.text_area(
    "Example prompts",
    value="""\t1. List the human genes encoding for the dishevelled proteins?\t2. What is the mode of inheritance of Facioscapulohumeral muscular dystrophy (FSHD)?\t3. What is HIV?""",
    height="content",
)
user_prompt = st.chat_input("Ask a medical question about the indexed corpus.")

if user_prompt:
    st.session_state.messages.append({"role": "user", "content": user_prompt})

    with st.chat_message("user"):
        st.markdown(user_prompt)

    with st.chat_message("assistant"):
        with st.spinner("Searching the corpus and generating an answer..."):
            sources = retrieve_context(client, embedder, collection_name, user_prompt, top_k)
            retrieval_stats = compute_retrieval_metrics(sources)
            prompt = build_prompt(user_prompt, sources)
            answer = st.write_stream(
                stream_answer_from_endpoint(
                    generation_url=generation_url,
                    prompt=prompt,
                    max_length=max_length,
                )
            )

        if not isinstance(answer, str):
            answer = str(answer)
        print("Answer: ", answer)

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "Avg Retrieval Score",
                f"{retrieval_stats['avg']:.3f}"
            )

        with col2:
            st.metric(
                "Best Retrieval Score",
                f"{retrieval_stats['max']:.3f}"
            )

        with st.expander("Retrieved context", expanded=False):
            render_sources(sources)
    st.session_state.messages.append({"role": "assistant", "content": answer})
