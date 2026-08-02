import re

from src.config import DEFAULT_MAX_NEW_TOKENS, DEFAULT_MODEL_NAME

SECTION_LABELS = ["SUBJECT", "GREETING", "BODY", "RECOMMENDATION", "OFFER", "CTA"]

FALLBACK_TEXT = {
    "SUBJECT": "Something we picked out for you",
    "GREETING": "Hi there,",
    "BODY": "We thought of you when we saw this - hope you like it as much as we do.",
    "RECOMMENDATION": "We think this could be a great fit based on what you've bought before.",
    "OFFER": "Enjoy a special discount on us.",
    "CTA": "Shop now",
}


class EmailGenerator:

    def __init__(self, model_name: str = DEFAULT_MODEL_NAME, max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS):
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self._pipe = None

    def _load_pipeline(self):
        if self._pipe is not None:
            return self._pipe

        import torch
        from transformers import pipeline

        device_kwargs = {"device_map": "auto"} if torch.cuda.is_available() else {}

        try:
            self._pipe = pipeline(
                "text-generation",
                model=self.model_name,
                dtype="auto",
                **device_kwargs,
            )
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load model '{self.model_name}'. Make sure you have internet access for the "
                "first download, or try a smaller instruct model if this one is too big for your machine."
            ) from exc

        return self._pipe

    def generate_raw(self, prompt: str) -> str:
        pipe = self._load_pipeline()
        messages = [{"role": "user", "content": prompt}]

        try:
            output = pipe(
                messages,
                max_new_tokens=self.max_new_tokens,
                do_sample=True,
                temperature=0.8,
                top_p=0.9,
                pad_token_id=pipe.tokenizer.eos_token_id,
            )
        except Exception as exc:
            raise RuntimeError("Text generation failed midway through a request.") from exc

        reply = output[0]["generated_text"]
        if isinstance(reply, list):
            reply = reply[-1]["content"]

        return reply.strip()


def parse_email_sections(raw_text: str) -> dict:
    label_pattern = "|".join(SECTION_LABELS)
    pattern = rf"(?:^|\n)\s*({label_pattern})\s*:\s*(.*?)(?=\n\s*(?:{label_pattern})\s*:|\Z)"

    matches = re.findall(pattern, raw_text, re.DOTALL | re.IGNORECASE)

    sections = {}
    for label, content in matches:
        cleaned = " ".join(content.split())
        if cleaned:
            sections[label.upper()] = cleaned

    for label in SECTION_LABELS:
        if label not in sections:
            sections[label] = FALLBACK_TEXT[label]

    return sections


def generate_email_for_customer(generator, prompt):
    raw_reply = generator.generate_raw(prompt)
    return parse_email_sections(raw_reply)
