from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass
from typing import Any, Callable, Iterable

from .config import get_settings
from .prompts import get_prompt
from .services import build_career_consultation, build_recommendations, extract_cv_profile_data
from .tools import extract_target_role, rag_search_jobs, route_task, sql_query_jobs

logger = logging.getLogger(__name__)

try:
    from langchain.agents import create_agent
    from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
    from langchain_openai import ChatOpenAI
except Exception:
    create_agent = None
    ChatOpenAI = None
    ToolMessage = None
    HumanMessage = None
    SystemMessage = None


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


def _normalize_history_input(history: Any) -> list[dict[str, str]]:
    if isinstance(history, str):
        stripped = history.strip()
        return [{'role': 'system', 'content': stripped}] if stripped else []

    normalized: list[dict[str, str]] = []
    if isinstance(history, Iterable):
        for item in history:
            if not isinstance(item, dict):
                continue
            role = str(item.get('role', 'user')).strip() or 'user'
            content = str(item.get('content', '')).strip()
            if content:
                normalized.append({'role': role, 'content': content})
    return normalized


def _history_to_text(history: Any, limit: int = 8) -> str:
    normalized = _normalize_history_input(history)
    recent = normalized[-limit:]
    if not recent:
        return ''
    return '\n'.join(f"{item['role']}: {item['content']}" for item in recent)


def _estimate_tokens_from_text(*parts: Any) -> int:
    joined = ' '.join(str(part) for part in parts if part)
    if not joined:
        return 0
    return max(1, math.ceil(len(joined) / 4))


def _build_writer_messages(query: str, raw_result: str, history_text: str, intent: str):
    if HumanMessage is None or SystemMessage is None:
        return None
    prompt = get_prompt('natural_response_writer')
    content = (
        f'Intent: {intent}\n\n'
        f'Riwayat singkat:\n{history_text or "(tidak ada)"}\n\n'
        f'Pertanyaan user:\n{query}\n\n'
        f'Hasil tool / data kerja:\n{raw_result}\n\n'
        'Tulis jawaban final untuk user dalam bahasa Indonesia yang natural dan langsung menjawab kebutuhan user.'
    )
    return [SystemMessage(content=prompt), HumanMessage(content=content)]


def _extract_usage(response: Any) -> tuple[int, int]:
    usage = getattr(response, 'response_metadata', None) or {}
    token_usage = usage.get('token_usage') or usage.get('usage_metadata') or {}
    input_tokens = int(token_usage.get('prompt_tokens', token_usage.get('input_tokens', 0)) or 0)
    output_tokens = int(token_usage.get('completion_tokens', token_usage.get('output_tokens', 0)) or 0)
    return input_tokens, output_tokens


def _fallback_rag_narrative(text: str) -> str:
    if 'Result 1' not in text:
        return text
    blocks = [block.strip() for block in text.split('\n\n') if block.strip()]
    items: list[dict[str, str]] = []
    for block in blocks[:5]:
        fields: dict[str, str] = {}
        for line in block.splitlines():
            if ':' in line:
                key, value = line.split(':', 1)
                fields[key.strip().lower()] = value.strip()
        items.append(fields)

    if not items:
        return text

    intro = 'Saya menemukan beberapa lowongan yang paling relevan untuk pertanyaan Anda:'
    lines = [intro]
    for idx, item in enumerate(items, start=1):
        title = item.get('title', 'N/A')
        company = item.get('company', 'N/A')
        location = item.get('location', 'N/A')
        work_type = item.get('work_type', 'N/A')
        snippet = item.get('snippet', '').strip()
        summary = f'{idx}. **{title}** di **{company}** — {location} ({work_type})'
        if snippet:
            summary += f'\n   Cocok karena deskripsinya menyinggung: {snippet[:180]}...'
        lines.append(summary)
    lines.append('Kalau Anda mau, saya bisa lanjut bantu pilihkan yang paling cocok berdasarkan skill atau lokasi yang Anda incar.')
    return '\n'.join(lines)


def _fallback_sql_narrative(sql_output: str) -> str:
    try:
        payload = json.loads(sql_output)
    except Exception:
        return f'Berikut hasil analitik yang saya temukan:\n\n{sql_output}'

    mode = payload.get('mode', 'analitik')
    sql = payload.get('sql', '')
    rows = payload.get('rows', []) if isinstance(payload.get('rows'), list) else []
    lines = [f'Saya sudah cek data lowongan dengan mode **{mode}**.']
    if rows:
        lines.append('Ringkasan hasilnya:')
        for idx, row in enumerate(rows[:5], start=1):
            if isinstance(row, dict):
                pretty = ', '.join(f'{key}: {value}' for key, value in row.items())
            else:
                pretty = str(row)
            lines.append(f'{idx}. {pretty}')
    else:
        lines.append('Saya belum menemukan baris hasil yang cukup untuk disimpulkan.')
    if sql:
        lines.append(f'Query yang dipakai: `{sql}`')
    return '\n'.join(lines)


def _fallback_cv_narrative(profile: dict[str, Any]) -> str:
    skills = ', '.join(profile.get('skills', [])[:8]) or 'belum terdeteksi'
    roles = ', '.join(profile.get('likely_roles', [])[:5]) or 'belum terdeteksi'
    experience = profile.get('estimated_years_experience', 0) or 0
    headline = profile.get('headline', '')
    lines = ['Saya sudah merangkum CV Anda.']
    if headline:
        lines.append(f'- Ringkasan profil: {headline}')
    lines.append(f'- Skill yang paling terlihat: {skills}')
    lines.append(f'- Role yang kemungkinan cocok: {roles}')
    if experience:
        lines.append(f'- Estimasi pengalaman yang terdeteksi: sekitar {experience} tahun')
    return '\n'.join(lines)


def _fallback_consultation_narrative(consultation: dict[str, Any]) -> str:
    matched = ', '.join(consultation.get('matched_skills', [])[:6]) or 'belum ada yang terdeteksi kuat'
    missing = ', '.join(consultation.get('missing_skills', [])[:6]) or 'tidak ada gap utama yang menonjol'
    recommendations = consultation.get('recommendations', [])
    market = consultation.get('market_summary', {}) if isinstance(consultation.get('market_summary'), dict) else {}
    lines = [f"Untuk target role **{consultation.get('target_role', 'N/A')}**, posisi Anda saat ini cukup seperti ini:"]
    lines.append(f'- Skill yang sudah selaras: {matched}')
    lines.append(f'- Skill yang masih perlu diperkuat: {missing}')
    sample_titles = market.get('sample_titles') or []
    if sample_titles:
        lines.append(f'- Contoh role di pasar: {", ".join(sample_titles[:3])}')
    if recommendations:
        lines.append('Langkah berikut yang saya sarankan:')
        for item in recommendations[:3]:
            lines.append(f'- {item}')
    return '\n'.join(lines)


def _compose_natural_answer(query: str, raw_result: str, history: Any, intent: str) -> dict[str, Any]:
    history_text = _history_to_text(history)
    estimated_input_tokens = _estimate_tokens_from_text(get_prompt('natural_response_writer'), intent, history_text, query, raw_result)
    settings = get_settings()
    if settings.openai_api_key and ChatOpenAI is not None:
        try:
            model = ChatOpenAI(
                model=settings.llm_model,
                openai_api_key=settings.openai_api_key,
                temperature=0.7,
            )
            messages = _build_writer_messages(query, raw_result, history_text, intent)
            if messages is not None:
                response = model.invoke(messages)
                text = response.content if hasattr(response, 'content') else str(response)
                input_tokens, output_tokens = _extract_usage(response)
                return {
                    'response': text,
                    'input_tokens': input_tokens or estimated_input_tokens,
                    'output_tokens': output_tokens or _estimate_tokens_from_text(text),
                    'token_mode': 'provider_usage' if input_tokens or output_tokens else 'estimated',
                }
        except Exception as exc:
            logger.warning('Natural response writer fallback activated: %s', exc)

    if intent == 'sql':
        text = _fallback_sql_narrative(raw_result)
    elif intent == 'cv':
        try:
            text = _fallback_cv_narrative(json.loads(raw_result))
        except Exception:
            text = raw_result
    elif intent == 'consultation':
        try:
            text = _fallback_consultation_narrative(json.loads(raw_result))
        except Exception:
            text = raw_result
    else:
        text = _fallback_rag_narrative(raw_result)

    return {
        'response': text,
        'input_tokens': estimated_input_tokens,
        'output_tokens': _estimate_tokens_from_text(text),
        'token_mode': 'estimated',
    }


def _format_rag_answer(text: str) -> str:
    return _fallback_rag_narrative(text)


def local_chat_response(query: str, history: Any = '') -> dict[str, Any]:
    used_tools: list[str] = []
    tool_messages: list[str] = []
    input_tokens = 0
    output_tokens = 0

    intent_payload = route_task.invoke({'query': query})
    intent = json.loads(intent_payload)['intent']
    used_tools.append('route_task')
    tool_messages.append(intent_payload)

    if intent == 'hybrid':
        rag_output = rag_search_jobs.invoke({'query': query, 'k': 3})
        sql_output = sql_query_jobs.invoke({'question': query})
        used_tools.extend(['rag_search_jobs', 'sql_query_jobs'])
        tool_messages.extend([str(rag_output), str(sql_output)])
        raw_answer = (
            'Hasil pencarian lowongan:\n'
            f'{_format_rag_answer(str(rag_output))}\n\n'
            'Hasil analitik data:\n'
            f'{_fallback_sql_narrative(str(sql_output))}'
        )
        composed = _compose_natural_answer(query, raw_answer, history, intent)
    elif intent == 'sql':
        sql_output = sql_query_jobs.invoke({'question': query})
        used_tools.append('sql_query_jobs')
        tool_messages.append(str(sql_output))
        composed = _compose_natural_answer(query, str(sql_output), history, intent)
    elif intent == 'cv':
        text = _history_to_text(history, limit=12) if _history_to_text(history, limit=12) else query
        profile = extract_cv_profile_data(text)
        used_tools.append('extract_cv_profile')
        tool_messages.append(str(profile))
        composed = _compose_natural_answer(query, json.dumps(profile, ensure_ascii=False), history, intent)
    elif intent == 'consultation':
        text = _history_to_text(history, limit=12) if _history_to_text(history, limit=12) else query
        target_role = extract_target_role(query) or 'Data Analyst'
        consultation = build_career_consultation(text, target_role)
        used_tools.append('analyze_skill_gap')
        tool_messages.append(str(consultation))
        composed = _compose_natural_answer(query, json.dumps(consultation, ensure_ascii=False), history, intent)
    else:
        rag_output = rag_search_jobs.invoke({'query': query, 'k': 5})
        used_tools.append('rag_search_jobs')
        tool_messages.append(str(rag_output))
        raw_answer = _format_rag_answer(str(rag_output))
        if 'cv' in query.lower() or 'resume' in query.lower():
            rec = build_recommendations(query, top_k=3)
            if rec['matches']:
                used_tools.append('build_recommendations')
                tool_messages.append(str(rec))
                top_match = rec['matches'][0]
                raw_answer += (
                    '\n\nTambahan rekomendasi cepat:\n'
                    f"Lowongan yang paling menonjol adalah {top_match.get('job_title', 'N/A')} di {top_match.get('company_name', 'N/A')}."
                )
        composed = _compose_natural_answer(query, raw_answer, history, intent)

    input_tokens += composed.get('input_tokens', 0)
    output_tokens += composed.get('output_tokens', 0)

    return {
        'response': composed['response'],
        'input_tokens': input_tokens,
        'output_tokens': output_tokens,
        'tool_messages': tool_messages,
        'used_tools': used_tools,
        'token_mode': composed.get('token_mode', 'estimated'),
    }


def _build_langchain_supervisor():
    settings = get_settings()
    if not settings.openai_api_key or create_agent is None or ChatOpenAI is None:
        return None
    try:
        model = ChatOpenAI(model=settings.llm_model, openai_api_key=settings.openai_api_key, temperature=0.2)
        from .tools import tool

        @tool
        def call_rag_agent(query: str, history: str = '') -> str:
            return local_chat_response(query, history)['response']

        @tool
        def call_sql_agent(query: str, history: str = '') -> str:
            return _fallback_sql_narrative(str(sql_query_jobs.invoke({'question': query})))

        @tool
        def call_cv_agent(query: str, history: str = '') -> str:
            profile = extract_cv_profile_data(history or query)
            return _fallback_cv_narrative(profile)

        @tool
        def call_consultation_agent(query: str, history: str = '') -> str:
            consultation = build_career_consultation(history or query, extract_target_role(query) or 'Data Analyst')
            return _fallback_consultation_narrative(consultation)

        return create_agent(
            model=model,
            tools=[call_rag_agent, call_sql_agent, call_cv_agent, call_consultation_agent],
            system_prompt=get_prompt('job_supervisor_agent'),
        )
    except Exception as exc:
        logger.warning('Failed to initialize langchain supervisor: %s', exc)
        return None


supervisor_agent = _build_langchain_supervisor()
