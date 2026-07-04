# 📄 Retail PDF Q&A Chatbot

A Retrieval-Augmented Generation(RAG) chatbot that answers questions about a specific PDF document, built with FAISS for semantic search and OpenAI's GPT-4o-mini for answer generation. Frontend built with Streamlit.

## Features

- Extracts and chunks text from a target PDF (`retail.pdf`)
- Generates semantic embeddings using `sentence-transformers` (`all-MiniLM-L6-v2`)
- Fast vector similarity search via FAISS
- Context-grounded answers using OpenAI's `gpt-4o-mini`
- Interactive Streamlit UI with chat history and source chunk transparency

## Tech Stack

- **PDF Parsing:** pypdf
- **Embeddings:** sentence-transformers
- **Vector Search:** FAISS
- **LLM:** OpenAI GPT-4o-mini
- **Frontend:** Streamlit

## Project Structure
chatbot-tredence/
├── app.py # Streamlit app (main entry point)
├── retail.pdf # Source document
├── requirements.txt # Python dependencies
├── .env # API keys (not committed)
├── .gitignore
└── README.md


## Setup

1. Clone the repository:
```bash
git clone https://github.com/<your-username>/chatbot-tredence.git
cd chatbot-tredence
```

2. Create and activate a virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Add your OpenAI API key:
Create a `.env` file in the project root:
OPENAI_API_KEY=your_openai_key_here

## Usage

Run the Streamlit app:
```bash
streamlit run app.py
```

Open the browser tab (usually `http://localhost:8501`) and start asking questions about `retail.pdf`.

## How It Works

1. **Ingestion:** The PDF is parsed page by page and cleaned of whitespace/formatting noise.
2. **Chunking:** Text is split into overlapping chunks (1200 chars, 200 overlap) to preserve context across boundaries.
3. **Embedding:** Each chunk is converted into a vector using a sentence-transformer model.
4. **Retrieval:** User queries are embedded and matched against chunk vectors using FAISS (L2 distance).
5. **Generation:** Top-k relevant chunks are passed as context to GPT-4o-mini, which generates a grounded answer.

## Future Improvements

- Support for multiple PDF uploads
- Persistent vector store (e.g., ChromaDB) instead of local FAISS files
- Answer confidence scoring
- Deployment to Streamlit Community Cloud

## License

MIT
