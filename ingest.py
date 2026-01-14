import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

VECTOR_PATH = "data/vectorstore"

def ingest_pdfs(pdf_files):
    documents = []

    for pdf in pdf_files:
        loader = PyPDFLoader(pdf)
        docs = loader.load()

        for d in docs:
            d.metadata["source"] = os.path.basename(pdf)
        documents.extend(docs)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=200
    )

    chunks = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    db = FAISS.from_documents(chunks, embeddings)
    db.save_local(VECTOR_PATH)

    return db
