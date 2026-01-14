from langchain_core.prompts import PromptTemplate
   

QA_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are a strict AI assistant.

Answer the question ONLY using the context below.
If the answer is not present, say:
"Answer not found in the provided documents."

Context:
{context}

Question:
{question}

Rules:
- Do not use outside knowledge
- Cite document name and page number
"""
)

VERIFY_PROMPT = PromptTemplate(
    input_variables=["answer", "context"],
    template="""
You are a verifier AI.

Check if EVERY claim in the answer is supported by the context.
Reply only in JSON:

{{
  "verdict": "PASS" or "FAIL",
  "reason": "short explanation"
}}

Answer:
{answer}

Context:
{context}
"""
)
