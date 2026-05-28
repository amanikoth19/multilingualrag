from langchain_ollama import OllamaLLM
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough


LANGUAGES = {
    "English": "English",
    "Hindi": "Hindi (हिंदी)",
    "Telugu": "Telugu (తెలుగు)",
    "Spanish": "Spanish (Español)",
    "French": "French (Français)",
}


def get_qa_chain(response_language: str = "English", model: str = "llama3.2"):
    
    #load embeddings model
    embeddings = SentenceTransformerEmbeddings(
        model_name="paraphrase-multilingual-MiniLM-L12-v2"
    )

    #load chromadb vectorstore
    vectorstore = Chroma(
        persist_directory="./db",      
        embedding_function=embeddings  
    )
    
    #retriever — fetches top 4 similar chunks
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    #LLMs
    llm = OllamaLLM(model=model, temperature=0.2)
    
    #prompt template
    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="""You are a helpful multilingual assistant.
Use ONLY the context below to answer the question.
Always respond in {language}, regardless of what language the document or question is in.
If you don't know the answer from the context, say so in {language}.
Do not make up any information.

Context:
{{context}}

Question: {{question}}

Answer in {language}:""".format(language=response_language)
    )

    # helper to format retrieved docs into one string
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # modern langchain chain using LCEL (pipe syntax)
    chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain, retriever
