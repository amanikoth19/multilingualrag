from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings


def ingest_file(file_path: str):

    # choose loader
    if file_path.endswith(".pdf"):
        loader = PyPDFLoader(file_path)
    else:
        loader = TextLoader(file_path, encoding="utf-8")

    # load documents here
    docs = loader.load()

    print(f"📄 Loaded {len(docs)} page(s) from {file_path}")

    # split text into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=60
    )

    chunks = splitter.split_documents(docs)
    
    print(f"✂️ Split into {len(chunks)} chunks")
    
    if len(chunks) == 0:
        raise ValueError("No text chunks were created from the document.")

    # create embeddings
    embeddings = SentenceTransformerEmbeddings(
        model_name="paraphrase-multilingual-MiniLM-L12-v2"
    )

    # store in ChromaDB
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="./db"
    )

    print(f"✅ Stored {len(chunks)} chunks in ChromaDB")

    return vectorstore
