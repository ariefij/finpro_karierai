from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    query: str = Field(..., description='Pertanyaan user')
    history: str = Field(default='', description='Ringkasan chat sebelumnya')


class ChatResponse(BaseModel):
    response: str
    input_tokens: int = 0
    output_tokens: int = 0
    tool_messages: list[str] = Field(default_factory=list)
    used_tools: list[str] = Field(default_factory=list)


class IngestResponse(BaseModel):
    jobs_inserted: int
    chunks_inserted: int
    collection_name: str


class CVAnalyzeRequest(BaseModel):
    cv_text: str


class CVAnalyzeResponse(BaseModel):
    profile: dict[str, Any]


class RecommendationMatch(BaseModel):
    job_id: str | None = None
    job_title: str | None = None
    company_name: str | None = None
    location: str | None = None
    work_type: str | None = None
    salary_raw: str | None = None
    score: float = 0.0
    matched_skills: list[str] = Field(default_factory=list)
    explanation: list[str] = Field(default_factory=list)
    job_excerpt: str | None = None


class RecommendationRequest(BaseModel):
    cv_text: str
    top_k: int = Field(default=5, ge=1, le=20)


class RecommendationResponse(BaseModel):
    profile: dict[str, Any]
    search_query: str
    matches: list[RecommendationMatch] = Field(default_factory=list)


class ConsultationRequest(BaseModel):
    cv_text: str
    target_role: str


class ConsultationResponse(BaseModel):
    target_role: str
    profile: dict[str, Any]
    matched_skills: list[str] = Field(default_factory=list)
    missing_skills: list[str] = Field(default_factory=list)
    market_summary: dict[str, Any] = Field(default_factory=dict)
    recommendations: list[str] = Field(default_factory=list)


class RouteTaskInput(BaseModel):
    query: str


class RAGSearchInput(BaseModel):
    query: str
    k: int = Field(default=5, ge=1, le=20)


class SQLQuestionInput(BaseModel):
    question: str


class CVTextInput(BaseModel):
    cv_text: str


class SkillGapInput(BaseModel):
    cv_text: str
    target_role: str
