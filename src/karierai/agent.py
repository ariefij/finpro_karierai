from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, Callable

from .config import get_settings
from .prompts import get_prompt
from .services import build_career_consultation, build_recommendations, extract_cv_profile_data
from .telemetry import build_invoke_config
from .tools import extract_target_role, rag_search_jobs, route_task, sql_query_jobs

logger = logging.getLogger(__name__)

try:
    from langchain.agents import create_agent
    from langchain_openai import ChatOpenAI
    from langchain_core.messages import ToolMessage
except Exception:
    create_agent = None
    ChatOpenAI = None
    ToolMessage = None


@dataclass
class SimpleAgent:
    name: str
    tool_name: str
    runner: Callable[..., str]

    def invoke(self, **kwargs: Any) -> str:
        return self.runner(**kwargs)


rag_agent = SimpleAgent(
    'rag_agent',
    'rag_search_jobs',
    lambda query, history='': str(rag_search_jobs.invoke({'query': query, 'k': 5})),
)
sql_agent = SimpleAgent(
    'sql_agent',
    'sql_query_jobs',
    lambda query, history='': str(sql_query_jobs.invoke({'question': query})),
)
cv_analyzer_agent = SimpleAgent(
    'cv_analyzer_agent',
    'extract_cv_profile',
    lambda query, history='': json.dumps(extract_cv_profile_data(history or query), ensure_ascii=False),
)
career_consultant_agent = SimpleAgent(
    'career_consultant_agent',
    'analyze_skill_gap',
    lambda query, history='': json.dumps(
        build_career_consultation(history or query, extract_target_role(query) or 'Data Analyst'),
        ensure_ascii=False,
    ),
)


def _format_rag_answer(text: str) -> str:
    if 'Result 1' not in text:
        return text
    blocks = [block.strip() for block in text.split('\n\n') if block.strip()]
    summaries = []
    for block in blocks[:5]:
        fields = {}
        for line in block.splitlines():
            if ':' in line:
                key, value = line.split(':', 1)
                fields[key.strip().lower()] = value.strip()
        summaries.append(
            f"- {fields.get('title', 'N/A')} | {fields.get('company', 'N/A')} | {fields.get('location', 'N/A')} | {fields.get('work_type', 'N/A')}"
        )
    return 'Berikut lowongan yang paling relevan:\n' + '\n'.join(summaries)


def local_chat_response(query: str, history: str = '') -> dict[str, Any]:
    used_tools: list[str] = []
    tool_messages: list[str] = []

    intent_payload = route_task.invoke({'query': query})
    intent = json.loads(intent_payload)['intent']
    used_tools.append('route_task')
    tool_messages.append(intent_payload)

    if intent == 'hybrid':
        rag_output = rag_search_jobs.invoke({'query': query, 'k': 3})
        sql_output = sql_query_jobs.invoke({'question': query})
        used_tools.extend(['rag_search_jobs', 'sql_query_jobs'])
        tool_messages.extend([str(rag_output), str(sql_output)])
        response = _format_rag_answer(str(rag_output)) + '\n\nRingkasan SQL:\n' + str(sql_output)
    elif intent == 'sql':
        sql_output = sql_query_jobs.invoke({'question': query})
        used_tools.append('sql_query_jobs')
        tool_messages.append(str(sql_output))
        response = 'Hasil analitik SQL:\n' + str(sql_output)
    elif intent == 'cv':
        text = history if len(history.strip()) > 40 else query
        profile = extract_cv_profile_data(text)
        used_tools.append('extract_cv_profile')
        tool_messages.append(str(profile))
        skills = ', '.join(profile.get('skills', [])[:8]) or 'belum terdeteksi'
        roles = ', '.join(profile.get('likely_roles', [])[:5]) or 'belum terdeteksi'
        response = f'Ringkasan CV: skill utama {skills}. Role yang mungkin cocok: {roles}.'
    elif intent == 'consultation':
        text = history if len(history.strip()) > 40 else query
        target_role = extract_target_role(query) or 'Data Analyst'
        consultation = build_career_consultation(text, target_role)
        used_tools.append('analyze_skill_gap')
        tool_messages.append(str(consultation))
        matched = ', '.join(consultation['matched_skills'][:6]) or 'belum ada'
        missing = ', '.join(consultation['missing_skills'][:6]) or 'belum ada'
        recommendations = ' '.join(consultation['recommendations'][:2])
        response = (
            f"Untuk target role {consultation['target_role']}, skill yang sudah cocok: {matched}. "
            f"Skill yang masih perlu diperkuat: {missing}. {recommendations}"
        )
    else:
        rag_output = rag_search_jobs.invoke({'query': query, 'k': 5})
        used_tools.append('rag_search_jobs')
        tool_messages.append(str(rag_output))
        response = _format_rag_answer(str(rag_output))
        if 'cv' in query.lower() or 'resume' in query.lower():
            rec = build_recommendations(query, top_k=3)
            if rec['matches']:
                used_tools.append('build_recommendations')
                top_match = rec['matches'][0]
                response += (
                    '\n\nDari sudut pandang rekomendasi cepat, lowongan paling cocok saat ini adalah '
                    f"{top_match.get('job_title', 'N/A')} di {top_match.get('company_name', 'N/A')}."
                )

    return {
        'response': response,
        'input_tokens': 0,
        'output_tokens': 0,
        'tool_messages': tool_messages,
        'used_tools': used_tools,
    }


def _build_langchain_supervisor():
    settings = get_settings()
    if not settings.openai_api_key or create_agent is None or ChatOpenAI is None:
        return None
    try:
        model = ChatOpenAI(model=settings.llm_model, openai_api_key=settings.openai_api_key)
        from .tools import tool

        @tool
        def call_rag_agent(query: str, history: str = '') -> str:
            return local_chat_response(query, history)['response']

        @tool
        def call_sql_agent(query: str, history: str = '') -> str:
            return 'Hasil SQL:\n' + str(sql_query_jobs.invoke({'question': query}))

        @tool
        def call_cv_agent(query: str, history: str = '') -> str:
            return json.dumps(extract_cv_profile_data(history or query), ensure_ascii=False)

        @tool
        def call_consultation_agent(query: str, history: str = '') -> str:
            return json.dumps(
                build_career_consultation(history or query, extract_target_role(query) or 'Data Analyst'),
                ensure_ascii=False,
            )

        return create_agent(
            model=model,
            tools=[call_rag_agent, call_sql_agent, call_cv_agent, call_consultation_agent],
            system_prompt=get_prompt('job_supervisor_agent'),
        )
    except Exception as exc:
        logger.warning('Failed to initialize langchain supervisor: %s', exc)
        return None


supervisor_agent = _build_langchain_supervisor()
