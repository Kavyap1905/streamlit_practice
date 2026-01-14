import streamlit as st
import tempfile
from ingest import ingest_pdfs
from rag_chain import answer_with_verification
from dotenv import load_dotenv
load_dotenv()
import os
st.set_page_config(page_title="RAG Verifier", layout="wide")
st.title("Multi-Document RAG with Self-Verification")

uploaded_files = st.file_uploader(
    "Upload PDFs",
    type="pdf",
    accept_multiple_files=True
)

if uploaded_files:
    with st.spinner("Indexing documents..."):
        temp_paths = []
        for file in uploaded_files:
            temp = tempfile.NamedTemporaryFile(delete=False)
            temp.write(file.read())
            temp_paths.append(temp.name)

        ingest_pdfs(temp_paths)
        st.success("Documents indexed successfully!")

question = st.text_input("Ask a question")

if question:
    with st.spinner("Thinking..."):
        result = answer_with_verification(question)

    st.subheader("Answer")
    st.write(result["answer"])

    st.subheader("Verification Result")
    st.json(result["verification"])

    st.subheader("Confidence Score")
    st.progress(result["confidence"] / 100)
    st.write(f"**{result['confidence']}% confidence**")

    if result["confidence"] < 40:
        st.warning("Low confidence: answer may be incomplete or weakly supported.")
    elif result["confidence"] < 70:
        st.info("Moderate confidence: supported but limited evidence.")
    else:
        st.success("High confidence: well-supported by documents.")

    with st.expander("Sources"):
        for doc in result["sources"]:
            st.write(
                f"📄 {doc.metadata.get('source')} — Page {doc.metadata.get('page')}"
            )
            st.write(doc.page_content[:500] + "...")
