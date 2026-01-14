import json

def compute_confidence(answer, verification_json, docs):
    """
    Confidence is computed using:
    - Verification result
    - Number of retrieved sources
    - Whether model refused to hallucinate
    """

    # Safe default
    confidence = 50

    # If model clearly says answer not found
    if "not found in the provided documents" in answer.lower():
        return 0

    try:
        verification = json.loads(verification_json)
        verdict = verification.get("verdict", "FAIL")
    except Exception:
        verdict = "FAIL"

    num_sources = len(docs)

    if verdict == "FAIL":
        confidence = 30
    else:
        # PASS case
        confidence = 60

        if num_sources >= 3:
            confidence += 20
        elif num_sources == 2:
            confidence += 10

    return min(confidence, 100)
