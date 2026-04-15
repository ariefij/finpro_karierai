from __future__ import annotations

import json
import logging
import re
from functools import wraps
from typing import Any, Callable

from .database import fetch_job_by_id, get_vector_store, list_available_filters, run_safe_analytics, search_jobs
from .models import CVTextInput, RAGSearchInput, RouteTaskInput, SQLQuestionInput, SkillGapInput
from .services import build_career_consultation, extract_cv_profile_data

logger = logging.getLogger(__name__)


class SimpleTool:
    def __init__(self, func: Callable[..., Any], name: str | None = None):
        self.func = func
        self.name = name or func.__name__
        self.__name__ = func.__name__
        self.__doc__ = func.__doc__

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.func(*args, **kwargs)

    def invoke(self, input: Any) -> Any:
        if isinstance(input, dict):
            return self.func(**input)
        if isinstance(input, tuple):
            return self.func(*input)
        return self.func(input)


try:
    from langchain_core.tools import tool as _langchain_tool

    def tool(*args: Any, **kwargs: Any):
        return _langchain_tool(*args, **kwargs)
except Exception:

    def tool(*decorator_args: Any, **decorator_kwargs: Any):
        def decorator(func: Callable[..., Any]) -> SimpleTool:
            @wraps(func)
            def wrapped(*args: Any, **kwargs: Any) -> Any:
                return func(*args, **kwargs)

            return SimpleTool(wrapped, name=func.__name__)

        if decorator_args and callable(decorator_args[0]) and len(decorator_args) == 1 and not decorator_kwargs:
            return decorator(decorator_args[0])
        return decorator


HYBRID_PATTERNS = ['bandingkan', 'sekaligus jumlah', 'serta statistik', 'serta jumlah', 'top lokasi dan contoh']
SQL_PATTERNS = ['berapa', 'jumlah', 'terbanyak', 'distribusi', 'statistik', 'trend', 'tren']
CV_PATTERNS = ['cv', 'resume saya', 'profil saya', 'ringkas cv', 'analisis cv']
CONSULTATION_PATTERNS = ['gap skill', 'konsultasi', 'career', 'karier', 'cocok untuk role', 'role apa yang cocok']


def detect_intent(query: str) -> str:
    q = query.lower()
    if any(token in q for token in HYBRID_PATTERNS):
        return 'hybrid'
    if any(token in q for token in CONSULTATION_PATTERNS):
        return 'consultation'
    if any(token in q for token in CV_PATTERNS):
        return 'cv'
    if any(token in q for token in SQL_PATTERNS):
        return 'sql'
    return 'rag'


def extract_target_role(text: str) -> str | None:
    lower = text.lower()
    role_patterns = [
        r'(?:target role|posisi target|menjadi|jadi|role)\s*[:\-]?\s*([a-zA-Z ]{3,50})',
        r'(?:untuk|sebagai)\s+(data analyst|data scientist|business analyst|hr manager|recruiter)',
    ]
    for pattern in role_patterns:
        match = re.search(pattern, lower)
        if match:
            return match.group(1).strip().title()
    return None


def _format_search_rows(rows: list[dict], source: str) -> str:
    if not rows:
        return 'Tidak ada lowongan relevan ditemukan.'
    lines: list[str] = []
    for idx, row in enumerate(rows, start=1):
        lines.append(
            '\n'.join(
                [
                    f'Result {idx}',
                    f'source: {source}',
                    f"job_id: {row.get('job_id', 'N/A')}",
                    f"title: {row.get('job_title', 'N/A')}",
                    f"company: {row.get('company_name', 'N/A')}",
                    f"location: {row.get('location', 'N/A')}",
                    f"work_type: {row.get('work_type', 'N/A')}",
                    f"snippet: {str(row.get('job_description', ''))[:500]}",
                ]
            )
        )
    return '\n\n'.join(lines)


@tool(args_schema=RouteTaskInput)
def route_task(query: str) -> str:
    """Klasifikasikan intent user menjadi rag, sql, cv, consultation, atau hybrid."""
    return json.dumps({'intent': detect_intent(query)}, ensure_ascii=False)


@tool(args_schema=RAGSearchInput)
def rag_search_jobs(query: str, k: int = 5) -> str:
    """Cari lowongan paling relevan dari Qdrant. Fallback ke SQLite bila Qdrant tidak tersedia."""
    try:
        docs = get_vector_store().similarity_search_with_score(query, k=k)
        if docs:
            lines: list[str] = []
            for idx, (doc, score) in enumerate(docs, start=1):
                lines.append(
                    '\n'.join(
                        [
                            f'Result {idx}',
                            'source: qdrant',
                            f"job_id: {doc.metadata.get('job_id', 'N/A')}",
                            f"title: {doc.metadata.get('job_title', 'N/A')}",
                            f"company: {doc.metadata.get('company_name', 'N/A')}",
                            f"location: {doc.metadata.get('location', 'N/A')}",
                            f"work_type: {doc.metadata.get('work_type', 'N/A')}",
                            f'score: {score:.4f}',
                            f'snippet: {doc.page_content[:500]}',
                        ]
                    )
                )
            return '\n\n'.join(lines)
    except Exception as exc:
        logger.warning('Qdrant not available, fallback to SQLite lexical search: %s', exc)
    rows = search_jobs(search_query=query, limit=k)
    return _format_search_rows(rows, source='sqlite_fallback')


@tool(args_schema=SQLQuestionInput)
def sql_query_jobs(question: str) -> str:
    """Jawab pertanyaan terstruktur tentang lowongan menggunakan SQLite secara read-only."""
    return json.dumps(run_safe_analytics(question), ensure_ascii=False, indent=2)


@tool
def get_job_detail(job_id: str) -> str:
    """Ambil detail satu lowongan dari SQLite berdasarkan job_id."""
    row = fetch_job_by_id(job_id)
    return json.dumps(row, ensure_ascii=False, indent=2) if row else 'Job tidak ditemukan.'


@tool
def list_filters() -> str:
    """Ambil daftar lokasi, work type, dan company yang tersedia."""
    return json.dumps(list_available_filters(), ensure_ascii=False, indent=2)


@tool(args_schema=CVTextInput)
def extract_cv_profile(cv_text: str) -> str:
    """Ekstrak profil kandidat sederhana dari teks CV."""
    return json.dumps(extract_cv_profile_data(cv_text), ensure_ascii=False, indent=2)


@tool(args_schema=SkillGapInput)
def analyze_skill_gap(cv_text: str, target_role: str) -> str:
    """Analisis gap skill terhadap target role."""
    return json.dumps(build_career_consultation(cv_text, target_role), ensure_ascii=False, indent=2)
