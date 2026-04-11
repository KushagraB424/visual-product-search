"""
Shopping Agent — uses dynamically-generated search URLs and OpenRouter
to present shopping recommendations conversationally.

Key design change: The LLM does NOT invent URLs. Real search URLs are
constructed programmatically, then the LLM wraps them in a friendly message.
"""

from __future__ import annotations

import json
import logging
from typing import Dict, List, Optional
from urllib.parse import quote_plus

import httpx

from app.config import OPENROUTER_API_KEY, OPENROUTER_BASE_URL, OPENROUTER_MODEL

logger = logging.getLogger(__name__)


# ── Dynamic URL generation ───────────────────────────────────────────────

def generate_shopping_urls(
    label: str,
    color: Optional[str] = None,
    pattern: Optional[str] = None,
) -> List[Dict]:
    """
    Build real, working search URLs for major e-commerce platforms.
    These are search-page links that will always resolve.
    """
    parts = []
    if color and color not in ("unknown", "multicolor"):
        parts.append(color)
    if pattern and pattern not in ("solid", "unknown"):
        parts.append(pattern)
    parts.append(label)
    query = " ".join(parts)
    encoded = quote_plus(query)

    return [
        {
            "title": f"{query.title()} — Amazon",
            "url": f"https://www.amazon.in/s?k={encoded}",
            "platform": "Amazon",
            "price_range": "varies",
        },
        {
            "title": f"{query.title()} — Myntra",
            "url": f"https://www.myntra.com/{label.replace(' ', '-')}?rawQuery={encoded}",
            "platform": "Myntra",
            "price_range": "varies",
        },
        {
            "title": f"{query.title()} — Flipkart",
            "url": f"https://www.flipkart.com/search?q={encoded}",
            "platform": "Flipkart",
            "price_range": "varies",
        },
        {
            "title": f"{query.title()} — AJIO",
            "url": f"https://www.ajio.com/search/?text={encoded}",
            "platform": "AJIO",
            "price_range": "varies",
        },
        {
            "title": f"{query.title()} — Zara",
            "url": f"https://www.zara.com/in/en/search?searchTerm={encoded}",
            "platform": "Zara",
            "price_range": "varies",
        },
    ]


# ── System prompt for conversational presentation ────────────────────────

_CHAT_SYSTEM_PROMPT = """\
You are a friendly, enthusiastic fashion shopping assistant called **StyleBot**.
The user uploaded a fashion image and our AI detected clothing items.

Your job is to present shopping recommendations in a warm, conversational tone.

RULES:
1. Greet the user briefly and tell them what we detected.
2. For each detected item, present the shopping links we found.
3. Use the EXACT URLs provided below — do NOT invent, modify, or hallucinate any URLs.
4. Format each link as a clickable markdown link: [Title](URL)
5. Add brief, helpful styling tips where appropriate.
6. Keep the response concise but engaging — no more than 3-4 sentences per item.
7. Use emojis sparingly for a friendly touch (🛍️ 👗 ✨).
8. End with a brief closing line inviting further questions.
"""


class ShoppingAgent:
    """Async agent that generates shopping URLs and uses OpenRouter for chat."""

    def __init__(
        self,
        api_key: str = OPENROUTER_API_KEY,
        model: str = OPENROUTER_MODEL,
    ):
        self.api_key = api_key
        self.model = model
        self.enabled = bool(api_key)
        if not self.enabled:
            logger.warning(
                "OPENROUTER_API_KEY not set — shopping agent is disabled."
            )

    # ------------------------------------------------------------------ #

    async def chat(
        self,
        items: List[Dict],
        user_message: Optional[str] = None,
    ) -> tuple:
        """
        Generate shopping links for all items and ask the LLM to
        present them conversationally.

        Returns:
            (message: str, all_links: List[Dict])
        """
        # 1. Generate real URLs for each item
        all_links = []
        items_summary = []

        for item in items:
            label = item.get("label", "garment")
            color = item.get("color")
            pattern = item.get("pattern")
            confidence = item.get("confidence", 0)

            links = generate_shopping_urls(label, color, pattern)
            all_links.extend(links)

            items_summary.append({
                "label": label,
                "color": color or "unknown",
                "pattern": pattern or "solid",
                "confidence": round(confidence, 2),
                "links": links,
            })

        # 2. If LLM is disabled, return a static message
        if not self.enabled:
            return self._static_message(items_summary), all_links

        # 3. Build LLM prompt with the real links
        items_context = json.dumps(items_summary, indent=2)

        user_content = f"""We detected the following items in the user's image:

{items_context}

{f'User says: "{user_message}"' if user_message else 'Present the shopping recommendations for all detected items.'}"""

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": _CHAT_SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
            "temperature": 0.6,
            "max_tokens": 1500,
        }

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "http://localhost:8000",
            "X-OpenRouter-Title": "Fashion AI Shopping Assistant",
        }

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.post(
                    OPENROUTER_BASE_URL,
                    json=payload,
                    headers=headers,
                )
                resp.raise_for_status()
                data = resp.json()

            message = data["choices"][0]["message"]["content"].strip()
            return message, all_links

        except Exception as exc:
            logger.error("Shopping chat error: %s", exc)
            return self._static_message(items_summary), all_links

    # ------------------------------------------------------------------ #

    @staticmethod
    def _static_message(items_summary: List[Dict]) -> str:
        """Generate a static fallback message when the LLM is unavailable."""
        lines = ["✨ **Here's what we found in your image!**\n"]
        for item in items_summary:
            label = item["label"]
            color = item["color"]
            lines.append(f"**{color.title()} {label.title()}**:")
            for link in item["links"]:
                lines.append(f"  • [{link['title']}]({link['url']})")
            lines.append("")
        lines.append("Happy shopping! 🛍️ Let me know if you need anything else.")
        return "\n".join(lines)

    # ------------------------------------------------------------------ #
    # Legacy method — kept for backward compatibility

    async def find_products(
        self,
        label: str,
        color: Optional[str] = None,
        pattern: Optional[str] = None,
        attributes: Optional[Dict] = None,
    ) -> List[Dict]:
        """Generate shopping links (no LLM needed)."""
        return generate_shopping_urls(label, color, pattern)
