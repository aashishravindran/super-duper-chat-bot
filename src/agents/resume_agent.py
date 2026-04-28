"""
Resume rewrite agent: prompt-based, no MCP or tool loop.

Flow:
  1. Extract job description from the user message (plain text or URL)
  2. If URL detected, fetch and strip HTML to get plain text
  3. Load the full resume from disk
  4. LLM rewrites the resume tailored to the job description
"""
import re
from html.parser import HTMLParser
from pathlib import Path

import httpx
from langchain_core.messages import SystemMessage

from src.config import get_llm
from src.state import State

RESUME_PATH = Path("data/resume.pdf")

SYSTEM_PROMPT = """You are an expert resume writer. Tailor the candidate's resume for the job description below.

Rules:
- Never fabricate experience, metrics, or dates — only reorder and reframe what exists
- Mirror keywords and phrases from the job description naturally throughout
- Lead each bullet point with the most relevant impact for this role
- Keep the same sections and structure as the original resume
- Output the full rewritten resume in clean Markdown

Resume:
{resume}

Job Description:
{job_description}
"""

_URL_RE = re.compile(r"https?://\S+")


class _TextExtractor(HTMLParser):
    def __init__(self):
        super().__init__()
        self._chunks: list[str] = []
        self._skip = False

    def handle_starttag(self, tag, _attrs):
        if tag in ("script", "style", "nav", "header", "footer"):
            self._skip = True

    def handle_endtag(self, tag):
        if tag in ("script", "style", "nav", "header", "footer"):
            self._skip = False

    def handle_data(self, data):
        if not self._skip:
            stripped = data.strip()
            if stripped:
                self._chunks.append(stripped)

    def get_text(self) -> str:
        return "\n".join(self._chunks)


async def _fetch_url(url: str) -> str:
    async with httpx.AsyncClient(follow_redirects=True, timeout=15) as client:
        response = await client.get(url, headers={"User-Agent": "Mozilla/5.0"})
        response.raise_for_status()
    extractor = _TextExtractor()
    extractor.feed(response.text)
    return extractor.get_text()[:8000]


def _load_resume() -> str:
    from langchain_community.document_loaders import PyPDFLoader
    pages = PyPDFLoader(str(RESUME_PATH)).load()
    return "\n\n".join(p.page_content for p in pages)


async def resume_node(state: State) -> dict:
    user_message = state["messages"][-1].content

    url_match = _URL_RE.search(user_message)
    if url_match:
        job_description = await _fetch_url(url_match.group())
    else:
        job_description = user_message

    resume_text = _load_resume()
    prompt = SYSTEM_PROMPT.format(resume=resume_text, job_description=job_description)
    response = await get_llm().ainvoke([SystemMessage(content=prompt)])
    return {"messages": [response]}
