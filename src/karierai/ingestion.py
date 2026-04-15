from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from .config import get_settings
from .database import get_vector_store, init_sqlite, insert_chunks, insert_jobs


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open('r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _clean_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() == 'none':
        return None
    return ' '.join(text.split())


def _build_job_id(row: dict[str, Any]) -> str:
    seed = '|'.join(
        [
            _clean_text(row.get('job_title')) or '',
            _clean_text(row.get('company_name')) or '',
            _clean_text(row.get('location')) or '',
            _clean_text(row.get('_scrape_timestamp')) or '',
        ]
    )
    return hashlib.md5(seed.encode('utf-8')).hexdigest()


def normalize_job(row: dict[str, Any], source_file: str = 'jobs.jsonl') -> dict[str, Any]:
    now = datetime.now(timezone.utc).isoformat()
    return {
        'job_id': _build_job_id(row),
        'job_title': _clean_text(row.get('job_title')) or 'Untitled Job',
        'company_name': _clean_text(row.get('company_name')),
        'location': _clean_text(row.get('location')),
        'work_type': _clean_text(row.get('work_type')),
        'salary_raw': _clean_text(row.get('salary')),
        'job_description': _clean_text(row.get('job_description')) or '',
        'scrape_timestamp': _clean_text(row.get('_scrape_timestamp')),
        'source_file': source_file,
        'created_at': now,
    }


def _chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    text = ' '.join(text.split())
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        start = max(0, end - chunk_overlap)
    return chunks


def build_chunk_rows(job: dict[str, Any]) -> list[dict[str, Any]]:
    settings = get_settings()
    header = (
        f"Job Title: {job.get('job_title') or '-'}\n"
        f"Company: {job.get('company_name') or '-'}\n"
        f"Location: {job.get('location') or '-'}\n"
        f"Work Type: {job.get('work_type') or '-'}\n"
        f"Salary: {job.get('salary_raw') or '-'}\n\nDescription:\n"
    )
    description_chunks = _chunk_text(job.get('job_description', ''), settings.chunk_size, settings.chunk_overlap)
    rows = []
    now = datetime.now(timezone.utc).isoformat()
    for idx, chunk in enumerate(description_chunks):
        chunk_text = header + chunk
        rows.append(
            {
                'chunk_id': f"{job['job_id']}-{idx}",
                'job_id': job['job_id'],
                'chunk_index': idx,
                'chunk_text': chunk_text,
                'char_count': len(chunk_text),
                'token_estimate': max(1, len(chunk_text) // 4),
                'created_at': now,
            }
        )
    return rows


def ingest_jobs(limit: int | None = None) -> dict[str, int | str]:
    settings = get_settings()
    raw_rows = load_jsonl(settings.jobs_path)
    if limit is not None:
        raw_rows = raw_rows[:limit]

    jobs = [normalize_job(row, source_file=settings.jobs_path.name) for row in raw_rows]
    chunk_rows = [chunk for job in jobs for chunk in build_chunk_rows(job)]

    init_sqlite()
    jobs_inserted = insert_jobs(jobs)
    chunks_inserted = insert_chunks(chunk_rows)

    try:
        from langchain_core.documents import Document
        documents = [
            Document(
                page_content=chunk['chunk_text'],
                metadata={**job, 'chunk_id': chunk['chunk_id'], 'chunk_index': chunk['chunk_index']},
            )
            for chunk in chunk_rows
            for job in jobs
            if job['job_id'] == chunk['job_id']
        ]
        get_vector_store().add_documents(documents=documents, ids=[str(uuid4()) for _ in documents])
        collection_name = settings.qdrant_collection_name
    except Exception:
        collection_name = settings.qdrant_collection_name

    return {
        'jobs_inserted': jobs_inserted,
        'chunks_inserted': chunks_inserted,
        'collection_name': collection_name,
    }
