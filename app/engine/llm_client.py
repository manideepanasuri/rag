import json
import logging
import time
import urllib.error
import urllib.request
from typing import Any

logger = logging.getLogger(__name__)


class GeminiClient:
    """
    Standard library REST client for Gemini text-embedding-004 and Gemini 1.5 Flash.
    Zero external SDK dependency.
    """

    def __init__(self, timeout: float = 7.0):
        self.timeout = timeout

    def get_embedding(self, text: str, api_key: str) -> list[float] | None:
        """
        Generate a 768-dim vector embedding using Google Gemini API.
        Tries gemini-embedding-001 (with 768 dim) then text-embedding-004.
        Returns None if API key is empty or request fails.
        """
        if not api_key or not text.strip():
            return None

        models_to_try = [
            ("gemini-embedding-001", {"outputDimensionality": 768}),
            ("text-embedding-004", {}),
        ]

        for model_name, extra_params in models_to_try:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:embedContent?key={api_key}"
            payload = {
                "model": f"models/{model_name}",
                "content": {
                    "parts": [{"text": text}]
                },
                **extra_params,
            }
            data = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(
                url,
                data=data,
                headers={"Content-Type": "application/json"},
                method="POST",
            )

            for attempt in range(2):
                try:
                    with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                        result = json.loads(resp.read().decode("utf-8"))
                        values = result.get("embedding", {}).get("values")
                        if values and isinstance(values, list):
                            return values
                except urllib.error.HTTPError as he:
                    if he.code == 429:
                        time.sleep(0.6 * (attempt + 1))
                        continue
                    break
                except Exception as e:
                    logger.debug("Gemini embedding model %s attempt %d failed: %s", model_name, attempt, e)
                    time.sleep(0.3)
                    continue

        logger.warning("All Gemini embedding model attempts failed")
        return None

    def generate_grounded_answer(
        self,
        question: str,
        context: str,
        api_key: str,
    ) -> str | None:
        """
        Generate a grounded natural answer using Gemini Flash.
        Strictly enforces the abstention rule: if context does not answer, returns
        'I don't have an answer for that.'
        """
        answer, _ = self.generate_grounded_answer_with_model(question, context, api_key)
        return answer

    def generate_grounded_answer_with_model(
        self,
        question: str,
        context: str,
        api_key: str,
    ) -> tuple[str | None, str | None]:
        """
        Generate grounded answer and return (answer_text, model_name).
        """
        if not api_key or not context.strip():
            return None, None

        prompt = (
            "You are a factual support assistant for Solstice Learning. "
            "Answer the question using ONLY the provided documentation context below. "
            "If the context does not completely answer the question, or if you are unsure, "
            "you MUST reply with the exact phrase: \"I don't have an answer for that.\" "
            "Never guess or extrapolate.\n\n"
            f"Context:\n{context}\n\n"
            f"Question:\n{question}\n\n"
            "Answer:"
        )

        payload = {
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": prompt}]
                }
            ],
            "generationConfig": {
                "temperature": 0.0,
                "maxOutputTokens": 200,
            }
        }
        data = json.dumps(payload).encode("utf-8")

        for gen_model in [
            "gemini-flash-lite-latest",
            "gemini-3.5-flash-lite",
            "gemini-3.1-flash-lite",
            "gemini-flash-latest",
        ]:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{gen_model}:generateContent?key={api_key}"
            req = urllib.request.Request(
                url,
                data=data,
                headers={"Content-Type": "application/json"},
                method="POST",
            )

            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    result = json.loads(resp.read().decode("utf-8"))
                    candidates = result.get("candidates", [])
                    if candidates:
                        parts = candidates[0].get("content", {}).get("parts", [])
                        if parts:
                            text = parts[0].get("text", "").strip()
                            return text, gen_model
            except Exception as e:
                logger.debug("Gemini generation model %s failed: %s", gen_model, e)
                continue

        logger.warning("All Gemini generation model attempts failed")
        return None, None
