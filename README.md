# Multilingual RAG System for Document Q&A
A multilingual Retrieval-Augmented Generation (RAG) 
system that lets you upload documents in any language and ask questions 
in your preferred language.

## Features
- Supports various languages such as Telugu, Hindi, Spanish, French
- Fully local — no data sent to cloud (uses Ollama + LLaMA 3.2)
- Multilingual retrieval — Telugu query finds English document chunks along with source citation
- Supports PDF and TXT file upload

## Tech Stack
| Component | Technology |
|---|---|
| LLM | Ollama (LLaMA 3.2) |
| Framework | LangChain |
| Embeddings | paraphrase-multilingual-MiniLM-L12-v2 |
| Vector DB | ChromaDB |
| UI | Streamlit |

## Architecture
Document (any language)
↓
Chunking (RecursiveCharacterTextSplitter)
↓
Multilingual Embeddings (Sentence Transformers)
↓
ChromaDB (Vector Storage)
↓
User Query (any language) → Embed → Search → Top 4 Chunks
↓
Ollama LLaMA 3.2 → Answer in chosen language

## How to Run

### Prerequisites
- Python 3.10+
- Ollama installed (ollama.com)

### Steps
```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/multilingual-doc-qa.git
cd multilingual-doc-qa

# 2. Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Mac/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Pull Ollama model
ollama pull llama3.2

# 5. Run the app
streamlit run app.py
```

### Usage
1. Open `http://localhost:8501`
2. Select reply language from sidebar
3. Upload a PDF or TXT document
4. Ask questions in any language required

## Project Structure

multilingual-doc-qa/
* app.py            # Streamlit UI  
* ingest.py         # Document loading, chunking, embedding  
* rag_chain.py      # RAG pipeline and LLM chain  
* requirements.txt  # Dependencies  
* README.md

## Example
- Upload an English government document
- Set reply language to Telugu
- Ask: రైతు బంధు పథకంలో ఎంత డబ్బు వస్తుంది?
- Get answer in Telugu 

##  Author
Aditi Manikoth
[LinkedIn](https://www.linkedin.com/in/aditi-manikoth-88b5001b9) | 
[GitHub](https://github.com/amanikoth19)
