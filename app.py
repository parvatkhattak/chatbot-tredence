import os
import re
import json
import numpy as np
import faiss
import streamlit as st
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

PDF_PATH = "retail.pdf"

def extract_text(pdf_path):
    reader = PdfReader(pdf_path)
    return [p.extract_text() for p in reader.pages if p.extract_text()]

def clean_text(text):
    return re.sub(r'\s+', ' ', text).strip()

def chunk_text(text, chunk_size=1200, overlap=200):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

@st.cache_resource
def load_embed_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

@st.cache_resource
def build_index(pdf_path):
    pages = extract_text(pdf_path)
    cleaned = [clean_text(p) for p in pages]
    full_text = " ".join(cleaned)
    chunks = chunk_text(full_text)

    embed_model = load_embed_model()
    embeddings = embed_model.encode(chunks, convert_to_numpy=True)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings.astype('float32'))
    return index, chunks

def retrieve_top_k(query, index, chunks, k=5):
    embed_model = load_embed_model()
    query_embedding = embed_model.encode([query], convert_to_numpy=True).astype('float32')
    _, indices = index.search(query_embedding, k)
    return [chunks[i] for i in indices[0]]

def ask_question(query, index, chunks, k=5):
    retrieved_chunks = retrieve_top_k(query, index, chunks, k=k)
    context = "\n\n".join(retrieved_chunks)

    prompt = f"""You are a helpful assistant answering questions based only on the provided document context.
If the answer isn't in the context, say you don't know.

Context:
{context}

Question: {query}

Answer:"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You answer questions using only the provided document context."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.2
    )
    return response.choices[0].message.content, retrieved_chunks
st.set_page_config(page_title="Retail PDF Q&A Bot", page_icon="📄", layout="wide")

st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #1e1e2f 0%, #2d1b4e 50%, #1e1e2f 100%);
}
h1 {
    background: linear-gradient(90deg, #ff6ec4, #7873f5, #4adede);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: 800;
    font-size: 3rem !important;
}
.stCaption, .st-emotion-cache-1629p8f {
    color: #cfcfea !important;
}
div[data-testid="stTextInput"] input {
    background-color: #2a2a40;
    color: #ffffff;
    border: 2px solid #7873f5;
    border-radius: 12px;
    padding: 10px;
}
.stButton button {
    background: linear-gradient(90deg, #ff6ec4, #7873f5);
    color: white;
    border: none;
    border-radius: 25px;
    padding: 10px 30px;
    font-weight: 700;
    font-size: 1rem;
    transition: transform 0.2s;
}
.stButton button:hover {
    transform: scale(1.05);
    box-shadow: 0 0 15px #7873f5;
}
.answer-box {
    background: linear-gradient(135deg, #2a2a40, #3a2a5a);
    border-left: 5px solid #4adede;
    border-radius: 12px;
    padding: 18px;
    margin-bottom: 10px;
    color: #f0f0ff;
}
.question-box {
    background: linear-gradient(135deg, #ff6ec4, #7873f5);
    border-radius: 12px;
    padding: 12px 18px;
    color: white;
    font-weight: 600;
    margin-bottom: 8px;
}
.stExpander {
    background-color: #23233a;
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)

st.title("📄 Retail PDF Q&A Bot")
st.caption("✨ Ask anything about retail.pdf — powered by GPT-4o-mini + semantic search")

with st.sidebar:
    st.markdown("## 🎯 About")
    st.info("This chatbot answers questions strictly from **retail.pdf** using retrieval-augmented generation (RAG).")
    st.markdown("## 💡 Try asking")
    sample_questions = [
        "What percentage of retail decisions are insights-informed?",
        "What is Tredence's 7-step approach?",
        "What role does GenAI play in retail transformation?",
        "What are the layers in the Agentic AI platform?"
    ]
    for sq in sample_questions:
        if st.button(sq, key=sq):
            st.session_state.pending_query = sq

index, chunks = build_index(PDF_PATH)

if "history" not in st.session_state:
    st.session_state.history = []
if "pending_query" not in st.session_state:
    st.session_state.pending_query = ""

col1, col2 = st.columns([5, 1])
with col1:
    query = st.text_input("Ask a question about the document:", value=st.session_state.pending_query, label_visibility="collapsed", placeholder="Type your question here...")
with col2:
    ask_clicked = st.button("🚀 Ask")

if ask_clicked and query:
    with st.spinner("🔍 Searching document and generating answer..."):
        answer, sources = ask_question(query, index, chunks)
    st.session_state.history.append((query, answer, sources))
    st.session_state.pending_query = ""

st.divider()

for q, a, sources in reversed(st.session_state.history):
    st.markdown(f'<div class="question-box">❓ {q}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="answer-box">💬 {a}</div>', unsafe_allow_html=True)
    with st.expander("📚 View source chunks used"):
        for i, s in enumerate(sources):
            st.markdown(f"**Chunk {i+1}:** {s}")
    st.write("")