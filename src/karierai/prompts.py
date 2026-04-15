from __future__ import annotations

import logging

from .telemetry import get_langfuse_client

logger = logging.getLogger(__name__)

PROMPT_FALLBACKS = {
    'job_supervisor_agent': (
        'You are the supervisor agent for KarierAI, an AI career assistant. '        'Decide whether the user needs rag, sql, cv, consultation, or hybrid processing. '        'Always prefer tools over guessing and answer in Indonesian.'
    ),
    'job_rag_agent': (
        'You are a RAG agent for job search. Use the retrieval tool to find relevant jobs and answer only from retrieved context.'
    ),
    'job_sql_agent': (
        'You are a SQL agent for structured job analytics. Use only read-only SQL style analytics and do not invent results.'
    ),
    'cv_analyzer_agent': (
        'You analyze CV text and extract skills, likely roles, education mentions, and experience evidence.'
    ),
    'career_consultant_agent': (
        'You provide career consultation using CV profile, target role, market demand, and skill-gap analysis.'
    ),
}


def get_prompt(prompt_name: str) -> str:
    client = get_langfuse_client()
    if client is not None:
        try:
            return client.get_prompt(prompt_name).get_langchain_prompt()
        except Exception as exc:
            logger.warning('Langfuse prompt load failed for %s: %s', prompt_name, exc)
    return PROMPT_FALLBACKS.get(prompt_name, 'You are a helpful assistant.')
