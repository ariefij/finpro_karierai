from __future__ import annotations

import json
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, Form, HTTPException, UploadFile

from .agent import ToolMessage, local_chat_response, supervisor_agent
from .config import get_settings
from .ingestion import ingest_jobs
from .models import (
    CVAnalyzeRequest,
    CVAnalyzeResponse,
    ChatRequest,
    ChatResponse,
    ConsultationRequest,
    ConsultationResponse,
    IngestResponse,
    RecommendationRequest,
    RecommendationResponse,
)
from .services import build_career_consultation, build_recommendations, extract_cv_profile_data, extract_text_from_upload_bytes
from .telemetry import build_invoke_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_ALLOWED_FILE_EXTENSIONS = {'.pdf', '.png', '.jpg', '.jpeg', '.webp', '.bmp', '.tif', '.tiff'}
_ALLOWED_CONTENT_TYPES = {
    'application/pdf',
    'application/x-pdf',
    'image/png',
    'image/jpeg',
    'image/jpg',
    'image/webp',
    'image/bmp',
    'image/tiff',
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info('Starting KarierAI service')
    yield
    logger.info('Stopping KarierAI service')


app = FastAPI(
    title='KarierAI API',
    description='Multi-agent API for Indonesian job dataset',
    lifespan=lifespan,
)


@app.get('/health')
def health() -> dict[str, str]:
    return {'status': 'ok'}


@app.get('/ready')
def ready() -> dict[str, object]:
    settings = get_settings()
    return {
        'sqlite_path': str(settings.sqlite_file),
        'jobs_path_exists': settings.jobs_path.exists(),
        'qdrant_configured': bool(settings.qdrant_url),
        'openai_configured': bool(settings.openai_api_key),
        'langfuse_configured': bool(settings.langfuse_public_key and settings.langfuse_secret_key),
    }


@app.post('/ingest', response_model=IngestResponse)
def ingest(limit: int | None = None) -> IngestResponse:
    try:
        return IngestResponse(**ingest_jobs(limit=limit))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post('/chat', response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    settings = get_settings()
    if supervisor_agent is None or not settings.openai_api_key:
        return ChatResponse(**local_chat_response(request.query, request.history))
    try:
        result = supervisor_agent.invoke(
            {'messages': [{'role': 'user', 'content': f"{request.query}\n\nHistory:\n{request.history}"}]},
            config=build_invoke_config({'route': '/chat'}),
        )
        messages = result['messages']
        response_text = messages[-1].content
        tool_messages: list[str] = []
        used_tools: list[str] = []
        input_tokens = 0
        output_tokens = 0
        for message in messages:
            if hasattr(message, 'response_metadata') and message.response_metadata:
                usage = message.response_metadata.get('token_usage') or message.response_metadata.get('usage_metadata') or {}
                input_tokens += usage.get('prompt_tokens', usage.get('input_tokens', 0))
                output_tokens += usage.get('completion_tokens', usage.get('output_tokens', 0))
            if ToolMessage is not None and isinstance(message, ToolMessage):
                tool_messages.append(str(message.content))
                if getattr(message, 'name', None):
                    used_tools.append(message.name)
        return ChatResponse(
            response=response_text,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            tool_messages=tool_messages,
            used_tools=used_tools,
        )
    except Exception as exc:
        logger.exception('chat failed, fallback to local orchestrator')
        local_result = local_chat_response(request.query, request.history)
        local_result['tool_messages'].append(f'llm_fallback={exc}')
        return ChatResponse(**local_result)


def _validate_upload(file: UploadFile) -> None:
    filename = (file.filename or '').lower()
    content_type = (file.content_type or '').lower()
    if any(filename.endswith(ext) for ext in _ALLOWED_FILE_EXTENSIONS):
        return
    if content_type in _ALLOWED_CONTENT_TYPES:
        return
    raise HTTPException(
        status_code=400,
        detail='Endpoint ini menerima CV dalam format PDF, PNG, JPG, JPEG, WEBP, BMP, atau TIFF.',
    )


async def _extract_cv_text_from_upload(file: UploadFile) -> str:
    _validate_upload(file)
    try:
        raw_bytes = await file.read()
    except Exception as exc:
        logger.exception('Gagal membaca file upload: %s', exc)
        raise HTTPException(status_code=400, detail=f'Gagal membaca file: {exc}') from exc

    if not raw_bytes:
        raise HTTPException(status_code=400, detail='File CV kosong.')
    try:
        return extract_text_from_upload_bytes(file.filename or '', file.content_type, raw_bytes)
    except ValueError as exc:
        logger.warning('CV processing error: %s', exc)
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception('Internal error during CV extraction: %s', exc)
        raise HTTPException(status_code=500, detail=f'Gagal memproses CV: {exc}') from exc


@app.post('/cv/analyze', response_model=CVAnalyzeResponse)
def cv_analyze(request: CVAnalyzeRequest) -> CVAnalyzeResponse:
    return CVAnalyzeResponse(profile=extract_cv_profile_data(request.cv_text))


@app.post('/cv/analyze-file', response_model=CVAnalyzeResponse)
async def cv_analyze_file(file: UploadFile = File(...)) -> CVAnalyzeResponse:
    cv_text = await _extract_cv_text_from_upload(file)
    return CVAnalyzeResponse(profile=extract_cv_profile_data(cv_text))


@app.post('/recommend', response_model=RecommendationResponse)
def recommend(request: RecommendationRequest) -> RecommendationResponse:
    return RecommendationResponse(**build_recommendations(request.cv_text, top_k=request.top_k))


@app.post('/recommend-file', response_model=RecommendationResponse)
async def recommend_file(file: UploadFile = File(...), top_k: int = Form(5)) -> RecommendationResponse:
    try:
        cv_text = await _extract_cv_text_from_upload(file)
        return RecommendationResponse(**build_recommendations(cv_text, top_k=top_k))
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception('Recommendation failed: %s', exc)
        raise HTTPException(status_code=500, detail=f'Gagal memberikan rekomendasi: {exc}') from exc


@app.post('/consult', response_model=ConsultationResponse)
def consult(request: ConsultationRequest) -> ConsultationResponse:
    return ConsultationResponse(**build_career_consultation(request.cv_text, request.target_role))


@app.post('/consult-file', response_model=ConsultationResponse)
async def consult_file(file: UploadFile = File(...), target_role: str = Form(...)) -> ConsultationResponse:
    cv_text = await _extract_cv_text_from_upload(file)
    return ConsultationResponse(**build_career_consultation(cv_text, target_role))


@app.get('/prompts/{prompt_name}')
def prompt_preview(prompt_name: str) -> dict[str, str]:
    from .prompts import get_prompt

    prompt = get_prompt(prompt_name)
    return {'name': prompt_name, 'prompt': prompt if isinstance(prompt, str) else json.dumps(prompt, ensure_ascii=False)}
