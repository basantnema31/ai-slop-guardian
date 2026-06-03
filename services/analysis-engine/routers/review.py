from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List
import os
from groq import Groq
import json

from cache.llm_cache import LLMCache

router = APIRouter()

# Module-level cache shared across all requests.
# max_size=256 entries, TTL=2 hours — review responses rarely need longer.
_review_cache: LLMCache = LLMCache(max_size=256, ttl_seconds=7200)


def _get_client() -> Groq | None:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return None
    return Groq(api_key=api_key)


class ReviewRequest(BaseModel):
    diff: str
    pr_title: str
    pr_body: str
    pr_labels: List[str] = []
    repo_id: str
    slop_score: float


@router.post("/generate")
async def generate_review(request: ReviewRequest):
    client = _get_client()
    if client is None:
        raise HTTPException(
            status_code=503,
            detail="GROQ_API_KEY is not configured",
        )
    labels_str = ", ".join(request.pr_labels) if request.pr_labels else "None"

    # --- Cache lookup -------------------------------------------------
    cache_key = LLMCache.make_key(
        request.diff[:6000],
        request.pr_title,
        request.pr_body,
        labels_str,
        model="llama-3.3-70b-versatile",
        slop_score=str(round(request.slop_score, 2)),
    )
    cached = _review_cache.get(cache_key)
    if cached is not None:
        return cached
    # ------------------------------------------------------------------

    system_prompt = (
        "You are a senior code reviewer for an open source project.\n\n"
        f"PR Context - Title: {request.pr_title} | "
        f"Labels: {labels_str}\n\n"
        "CRITICAL RULE:\n"
        "If the PR title starts with 'chore', 'refactor', or 'style', "
        "or contains linting-related labels, you must assume the removal "
        "of unused imports and type changes are intentional refactors. "
        "Do not flag them as Potential Bugs."
    )
    user_prompt = (
        "Review this PR diff. Be specific about:\n"
        "1) What the code actually does,\n"
        "2) Potential bugs or edge cases,\n"
        "3) Code quality issues,\n"
        "4) Suggested improvements.\n"
        "Be concise, technical, and helpful.\n\n"
        f"PR Title: {request.pr_title}\n"
        f"PR Body: {request.pr_body}\n"
        f"Slop Score: {request.slop_score}\n\n"
        "Diff:\n"
        f"{request.diff[:6000]}\n\n"
        "Return ONLY a JSON object: {\n"
        "  \"review\": \"detailed markdown review summary...\",\n"
        "  \"verdict\": \"approve\" | \"request_changes\" | \"comment\",\n"
        "  \"inline_comments\": [\n"
        "    {\"path\": \"path/to/file\", \"line\": 15, "
        "\"comment\": \"specific inline review comment...\"}\n"
        "  ]\n"
        "}\n"
        "Ensure \"review\" is a detailed markdown string with at least 2 "
        "paragraphs of analysis."
    )

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format={"type": "json_object"},
            max_tokens=1000,
        )

        content = response.choices[0].message.content.strip()
        # Robust JSON extraction
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()

        data = json.loads(content, strict=False)

        # --- Store result in cache before returning -------------------
        _review_cache.set(cache_key, data)
        # --------------------------------------------------------------

        return data
    except Exception as e:
        print(f"Review Generation Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/cache/stats")
async def review_cache_stats():
    """Return live statistics for the review LLM response cache."""
    return _review_cache.stats()


@router.delete("/cache")
async def clear_review_cache():
    """Flush all cached review responses (admin/debug use)."""
    _review_cache.clear()
    return {"message": "Review cache cleared"}
