from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.output_parsers import StrOutputParser

from prompts import QA_PROMPT, VERIFY_PROMPT
from confidence import compute_confidence

VECTOR_PATH = "data/vectorstore"


def load_vectorstore():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    return FAISS.load_local(
        VECTOR_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )


def get_llm():
    return ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0
    )


def answer_with_verification(question):
    db = load_vectorstore()
    retriever = db.as_retriever(search_kwargs={"k": 4})
    docs = retriever.invoke(question)

    context = "\n\n".join(
        f"{d.page_content}\n(Source: {d.metadata.get('source')}, Page: {d.metadata.get('page')})"
        for d in docs
    )

    llm = get_llm()

    qa_chain = QA_PROMPT | llm | StrOutputParser()
    answer = qa_chain.invoke({
        "context": context,
        "question": question
    })

    verify_chain = VERIFY_PROMPT | llm | StrOutputParser()
    verification = verify_chain.invoke({
        "answer": answer,
        "context": context
    })

    confidence = compute_confidence(answer, verification, docs)

    return {
        "answer": answer,
        "verification": verification,
        "confidence": confidence,
        "sources": docs
    }