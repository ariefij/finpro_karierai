from __future__ import annotations

import json
import re
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable

from .config import get_settings

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS jobs (
    job_id TEXT PRIMARY KEY,
    job_title TEXT NOT NULL,
    company_name TEXT,
    location TEXT,
    work_type TEXT,
    salary_raw TEXT,
    job_description TEXT NOT NULL,
    scrape_timestamp TEXT,
    source_file TEXT,
    created_at TEXT
);

CREATE TABLE IF NOT EXISTS job_chunks (
    chunk_id TEXT PRIMARY KEY,
    job_id TEXT NOT NULL,
    chunk_index INTEGER NOT NULL,
    chunk_text TEXT NOT NULL,
    char_count INTEGER,
    token_estimate INTEGER,
    created_at TEXT,
    FOREIGN KEY(job_id) REFERENCES jobs(job_id)
);
"""

SQLITE_WHITESPACE = re.compile(r'\s+')
SAFE_SQL_PREFIXES = ('select', 'with')
FORBIDDEN_SQL_TOKENS = re.compile(
    r'\b(insert|update|delete|drop|alter|create|replace|truncate|attach|detach|pragma|vacuum|reindex)\b',
    flags=re.IGNORECASE,
)
ALLOWED_TABLES = {'jobs', 'job_chunks'}
AGGREGATE_HINTS = [
    'jumlah', 'berapa', 'count', 'rata-rata', 'rata rata', 'average', 'avg',
    'tertinggi', 'terendah', 'minimum', 'maksimum', 'max', 'min', 'distribusi',
    'group by', 'per ', 'berdasarkan',
]
ROLE_HINTS = [
    'data analyst', 'data scientist', 'business analyst', 'product analyst', 'finance analyst',
    'recruiter', 'hr manager', 'machine learning engineer', 'business intelligence', 'talent acquisition',
]
LOCATION_STOPWORDS = {
    'yang', 'dengan', 'untuk', 'dan', 'atau', 'berapa', 'jumlah', 'rata', 'rata-rata',
    'salary', 'gaji', 'role', 'lowongan', 'job', 'jobs', 'full', 'part', 'time',
}
GROUP_DIMENSIONS = {
    'location': ['per lokasi', 'berdasarkan lokasi', 'by location', 'lokasi'],
    'company_name': ['per company', 'perusahaan', 'company', 'berdasarkan perusahaan', 'by company'],
    'work_type': ['per tipe kerja', 'per work type', 'work type', 'tipe kerja', 'jenis kerja'],
    'job_title': ['per role', 'per judul', 'judul lowongan', 'job title', 'role'],
    'source_file': ['per source', 'source file'],
}


def _normalize_salary_number(token: str, suffix: str) -> float | None:
    token = token.strip().replace(',', '.').replace(' ', '')
    if not token:
        return None
    if '.' in token and token.count('.') > 1 and all(len(part) == 3 for part in token.split('.')[1:]):
        numeric = float(token.replace('.', ''))
    elif ',' in token and token.count(',') > 1 and all(len(part) == 3 for part in token.split(',')[1:]):
        numeric = float(token.replace(',', ''))
    else:
        numeric = float(token)
    multiplier = {
        'm': 1_000_000,
        'jt': 1_000_000,
        'juta': 1_000_000,
        'k': 1_000,
        'rb': 1_000,
        'ribu': 1_000,
        '': 1,
    }.get(suffix.lower(), 1)
    return numeric * multiplier


def _extract_salary_numbers(value: str | None) -> list[float]:
    if not value:
        return []
    cleaned = str(value).lower().replace('\xa0', ' ')
    cleaned = cleaned.replace('idr', ' ').replace('rp', ' ')
    matches = re.findall(r'(\d+(?:[\.,]\d+)*)(?:\s*)(m|jt|juta|k|rb|ribu)?', cleaned)
    numbers: list[float] = []
    for token, suffix in matches:
        try:
            parsed = _normalize_salary_number(token, suffix)
        except Exception:
            parsed = None
        if parsed is not None:
            numbers.append(parsed)
    return numbers


def _salary_min(value: str | None) -> float | None:
    numbers = _extract_salary_numbers(value)
    return min(numbers) if numbers else None


def _salary_max(value: str | None) -> float | None:
    numbers = _extract_salary_numbers(value)
    return max(numbers) if numbers else None


def _salary_mid(value: str | None) -> float | None:
    numbers = _extract_salary_numbers(value)
    if not numbers:
        return None
    return sum(numbers) / len(numbers)


def _get_db_path() -> Path:
    settings = get_settings()
    path = settings.sqlite_file
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


@contextmanager
def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(_get_db_path())
    conn.row_factory = sqlite3.Row
    conn.create_function('salary_min', 1, _salary_min)
    conn.create_function('salary_max', 1, _salary_max)
    conn.create_function('salary_mid', 1, _salary_mid)
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_sqlite() -> None:
    with get_connection() as conn:
        conn.executescript(SCHEMA_SQL)


def insert_jobs(rows: Iterable[dict[str, Any]]) -> int:
    rows = list(rows)
    if not rows:
        return 0
    with get_connection() as conn:
        conn.executemany(
            """
            INSERT OR REPLACE INTO jobs (
                job_id, job_title, company_name, location, work_type,
                salary_raw, job_description, scrape_timestamp, source_file, created_at
            ) VALUES (
                :job_id, :job_title, :company_name, :location, :work_type,
                :salary_raw, :job_description, :scrape_timestamp, :source_file, :created_at
            )
            """,
            rows,
        )
    return len(rows)


def insert_chunks(rows: Iterable[dict[str, Any]]) -> int:
    rows = list(rows)
    if not rows:
        return 0
    with get_connection() as conn:
        conn.executemany(
            """
            INSERT OR REPLACE INTO job_chunks (
                chunk_id, job_id, chunk_index, chunk_text, char_count, token_estimate, created_at
            ) VALUES (
                :chunk_id, :job_id, :chunk_index, :chunk_text, :char_count, :token_estimate, :created_at
            )
            """,
            rows,
        )
    return len(rows)


def fetch_job_by_id(job_id: str) -> dict[str, Any] | None:
    init_sqlite()
    with get_connection() as conn:
        row = conn.execute('SELECT * FROM jobs WHERE job_id = ?', (job_id,)).fetchone()
    return dict(row) if row else None


def list_available_filters() -> dict[str, list[str]]:
    init_sqlite()
    with get_connection() as conn:
        locations = [r[0] for r in conn.execute('SELECT DISTINCT location FROM jobs WHERE location IS NOT NULL ORDER BY location LIMIT 20')]
        work_types = [r[0] for r in conn.execute('SELECT DISTINCT work_type FROM jobs WHERE work_type IS NOT NULL ORDER BY work_type LIMIT 20')]
        companies = [r[0] for r in conn.execute('SELECT company_name FROM jobs WHERE company_name IS NOT NULL GROUP BY company_name ORDER BY COUNT(*) DESC LIMIT 20')]
    return {'locations': locations, 'work_types': work_types, 'companies': companies}


def search_jobs(search_query: str = '', limit: int = 10) -> list[dict[str, Any]]:
    init_sqlite()
    search_query = SQLITE_WHITESPACE.sub(' ', search_query.strip())
    terms = [term for term in re.split(r'[,\s]+', search_query.lower()) if len(term) >= 2][:8]
    clauses = []
    params: list[Any] = []
    for term in terms:
        like_term = f'%{term}%'
        clauses.append('(LOWER(job_title) LIKE ? OR LOWER(company_name) LIKE ? OR LOWER(location) LIKE ? OR LOWER(work_type) LIKE ? OR LOWER(job_description) LIKE ?)')
        params.extend([like_term, like_term, like_term, like_term, like_term])

    sql = 'SELECT * FROM jobs'
    if clauses:
        sql += ' WHERE ' + ' AND '.join(clauses)
    sql += ' ORDER BY scrape_timestamp DESC, created_at DESC LIMIT ?'
    params.append(limit)

    with get_connection() as conn:
        rows = conn.execute(sql, tuple(params)).fetchall()
    return [dict(row) for row in rows]


def _find_role(question: str) -> str | None:
    q = question.lower()
    for role in ROLE_HINTS:
        if role in q:
            return role
    role_match = re.search(r'(?:role|posisi|lowongan|job)\s+([a-zA-Z ]{3,40})', q)
    if role_match:
        return ' '.join(role_match.group(1).split())
    return None


def _extract_phrase_after_markers(question: str, markers: list[str], stopwords: set[str]) -> str | None:
    q = question.lower()
    for marker in markers:
        match = re.search(rf'{marker}\s+([a-zA-Z][a-zA-Z\- ]{{2,50}})', q)
        if not match:
            continue
        phrase = ' '.join(match.group(1).split())
        words: list[str] = []
        for word in phrase.split():
            if word in stopwords:
                break
            words.append(word)
        if words:
            return ' '.join(words)
    return None


def _extract_filters(question: str) -> tuple[list[str], list[Any], dict[str, str]]:
    q = question.lower()
    clauses: list[str] = []
    params: list[Any] = []
    metadata: dict[str, str] = {}

    role = _find_role(q)
    if role:
        clauses.append('LOWER(job_title) LIKE ?')
        params.append(f'%{role}%')
        metadata['role'] = role

    location = _extract_phrase_after_markers(q, ['di', 'lokasi'], LOCATION_STOPWORDS)
    if location:
        clauses.append('LOWER(location) LIKE ?')
        params.append(f'%{location}%')
        metadata['location'] = location

    company = _extract_phrase_after_markers(q, ['company', 'perusahaan'], LOCATION_STOPWORDS)
    if company:
        clauses.append('LOWER(company_name) LIKE ?')
        params.append(f'%{company}%')
        metadata['company_name'] = company

    work_type_map = {
        'full time': '%full time%',
        'part time': '%part time%',
        'paruh waktu': '%paruh%',
        'intern': '%intern%',
        'internship': '%intern%',
        'magang': '%magang%',
        'contract': '%contract%',
        'kontrak': '%kontrak%',
        'remote': '%remote%',
        'hybrid': '%hybrid%',
    }
    for label, pattern in work_type_map.items():
        if label in q:
            clauses.append('LOWER(work_type) LIKE ?')
            params.append(pattern)
            metadata['work_type'] = label
            break

    if 'salary kosong' in q or 'gaji kosong' in q or 'salary missing' in q:
        clauses.append("(salary_raw IS NULL OR TRIM(LOWER(salary_raw)) IN ('', 'none'))")
        metadata['salary_state'] = 'missing'
    elif 'salary ada' in q or 'gaji ada' in q:
        clauses.append("(salary_raw IS NOT NULL AND TRIM(LOWER(salary_raw)) NOT IN ('', 'none'))")
        metadata['salary_state'] = 'available'

    return clauses, params, metadata


def _detect_group_by(question: str) -> str | None:
    q = question.lower()
    for column, hints in GROUP_DIMENSIONS.items():
        if any(hint in q for hint in hints):
            return column
    return None


def _is_listing_question(question: str) -> bool:
    q = question.lower()
    return any(token in q for token in ['cari', 'tampilkan', 'list', 'show', 'contoh lowongan', 'contoh job'])


def _build_local_sql(question: str) -> tuple[str, tuple[Any, ...], str]:
    q = question.lower().strip()
    clauses, params, metadata = _extract_filters(q)
    group_by = _detect_group_by(q)
    where_sql = f" WHERE {' AND '.join(clauses)}" if clauses else ''
    explanation = f'heuristic filters={metadata}'

    if any(token in q for token in ['rata-rata gaji', 'average salary', 'avg salary', 'rata rata gaji']):
        metric = 'AVG(salary_mid(salary_raw)) AS avg_salary'
        metric_order = 'avg_salary DESC'
        base_clause = '(salary_mid(salary_raw) IS NOT NULL)'
        where_sql = f' WHERE {base_clause}' + (f" AND {' AND '.join(clauses)}" if clauses else '')
        if group_by:
            sql = f'SELECT {group_by}, {metric} FROM jobs{where_sql} GROUP BY {group_by} ORDER BY {metric_order}'
        else:
            sql = f'SELECT {metric} FROM jobs{where_sql}'
        return sql, tuple(params), explanation

    if any(token in q for token in ['gaji tertinggi', 'salary tertinggi', 'max salary', 'maksimum gaji']):
        if group_by:
            sql = f'SELECT {group_by}, MAX(salary_max(salary_raw)) AS max_salary FROM jobs WHERE salary_max(salary_raw) IS NOT NULL'
            if clauses:
                sql += ' AND ' + ' AND '.join(clauses)
            sql += f' GROUP BY {group_by} ORDER BY max_salary DESC'
        else:
            sql = 'SELECT job_id, job_title, company_name, location, salary_raw, salary_max(salary_raw) AS max_salary FROM jobs WHERE salary_max(salary_raw) IS NOT NULL'
            if clauses:
                sql += ' AND ' + ' AND '.join(clauses)
            sql += ' ORDER BY max_salary DESC'
        return sql, tuple(params), explanation

    if any(token in q for token in ['gaji terendah', 'salary terendah', 'min salary', 'minimum gaji']):
        if group_by:
            sql = f'SELECT {group_by}, MIN(salary_min(salary_raw)) AS min_salary FROM jobs WHERE salary_min(salary_raw) IS NOT NULL'
            if clauses:
                sql += ' AND ' + ' AND '.join(clauses)
            sql += f' GROUP BY {group_by} ORDER BY min_salary ASC'
        else:
            sql = 'SELECT job_id, job_title, company_name, location, salary_raw, salary_min(salary_raw) AS min_salary FROM jobs WHERE salary_min(salary_raw) IS NOT NULL'
            if clauses:
                sql += ' AND ' + ' AND '.join(clauses)
            sql += ' ORDER BY min_salary ASC'
        return sql, tuple(params), explanation

    if any(token in q for token in ['perusahaan unik', 'company unik', 'distinct company']):
        return f'SELECT COUNT(DISTINCT company_name) AS total_companies FROM jobs{where_sql}', tuple(params), explanation

    if any(token in q for token in ['lokasi unik', 'distinct location']):
        return f'SELECT COUNT(DISTINCT location) AS total_locations FROM jobs{where_sql}', tuple(params), explanation

    if group_by and any(token in q for token in AGGREGATE_HINTS):
        return (
            f'SELECT {group_by}, COUNT(*) AS total FROM jobs{where_sql} GROUP BY {group_by} ORDER BY total DESC',
            tuple(params),
            explanation,
        )

    if any(token in q for token in ['jumlah', 'berapa banyak', 'count']) or ('berapa' in q and not _is_listing_question(q)):
        return f'SELECT COUNT(*) AS total FROM jobs{where_sql}', tuple(params), explanation

    if _is_listing_question(q):
        select_cols = 'job_id, job_title, company_name, location, work_type, salary_raw'
        order_by = 'ORDER BY scrape_timestamp DESC, created_at DESC'
        if 'gaji tertinggi' in q or 'salary tertinggi' in q:
            order_by = 'ORDER BY salary_max(salary_raw) DESC'
        return f'SELECT {select_cols} FROM jobs{where_sql} {order_by}', tuple(params), explanation

    sql = 'SELECT job_id, job_title, company_name, location, work_type, salary_raw FROM jobs'
    if where_sql:
        sql += where_sql
    sql += ' ORDER BY scrape_timestamp DESC, created_at DESC'
    return sql, tuple(params), explanation


def _build_schema_for_llm() -> str:
    return (
        'Gunakan SQLite. Tabel jobs(job_id, job_title, company_name, location, work_type, salary_raw, '
        'job_description, scrape_timestamp, source_file, created_at). '
        'Tabel job_chunks(chunk_id, job_id, chunk_index, chunk_text, char_count, token_estimate, created_at). '
        'Fungsi SQLite yang boleh dipakai: salary_min(salary_raw), salary_max(salary_raw), salary_mid(salary_raw). '
        'Balas JSON dengan kunci sql dan explanation. Hanya SELECT/CTE read-only.'
    )


def _generate_sql_with_llm(question: str) -> tuple[str, str] | None:
    settings = get_settings()
    if not settings.openai_api_key:
        return None
    try:
        from langchain_openai import ChatOpenAI
    except Exception:
        return None

    prompt = (
        f'{_build_schema_for_llm()}\n\n'
        'Aturan: gunakan hanya tabel/kolom yang tersedia, jangan gunakan DML/DDL, '
        'tambahkan LIMIT 50 jika query menampilkan daftar row.\n'
        f'Pertanyaan user: {question}\n\n'
        'Contoh output: {"sql": "SELECT COUNT(*) AS total FROM jobs", "explanation": "count all jobs"}'
    )
    model = ChatOpenAI(model=settings.llm_model, openai_api_key=settings.openai_api_key, temperature=0)
    response = model.invoke(prompt)
    content = response.content if hasattr(response, 'content') else str(response)
    text = content if isinstance(content, str) else json.dumps(content, ensure_ascii=False)
    match = re.search(r'\{.*\}', text, flags=re.DOTALL)
    if not match:
        return None
    payload = json.loads(match.group(0))
    sql = str(payload.get('sql', '')).strip()
    explanation = str(payload.get('explanation', 'llm'))
    if not sql:
        return None
    return sql, explanation


def _validate_sql(sql: str) -> str:
    normalized = ' '.join(sql.strip().split())
    if not normalized:
        raise ValueError('SQL kosong.')
    if ';' in normalized.rstrip(';'):
        raise ValueError('Hanya satu statement SQL yang diizinkan.')
    normalized = normalized.rstrip(';')
    if not normalized.lower().startswith(SAFE_SQL_PREFIXES):
        raise ValueError('Hanya query SELECT/CTE yang diizinkan.')
    if FORBIDDEN_SQL_TOKENS.search(normalized):
        raise ValueError('Query mengandung operasi yang tidak diizinkan.')
    tables = [match.group(1).lower() for match in re.finditer(r'\b(?:from|join)\s+([a-zA-Z_][\w]*)', normalized, flags=re.IGNORECASE)]
    if any(table not in ALLOWED_TABLES for table in tables):
        raise ValueError('Query mengakses tabel di luar whitelist.')
    if ' limit ' not in f' {normalized.lower()} ':
        normalized += ' LIMIT 50'
    return normalized


def _execute_safe_sql(sql: str, params: tuple[Any, ...] = ()) -> list[dict[str, Any]]:
    with get_connection() as conn:
        rows = conn.execute(sql, params).fetchall()
    return [dict(row) for row in rows]


def run_safe_analytics(question: str) -> dict[str, Any]:
    init_sqlite()
    llm_error = None
    llm_candidate = None
    try:
        llm_candidate = _generate_sql_with_llm(question)
    except Exception as exc:  # pragma: no cover - external dependency path
        llm_error = str(exc)

    if llm_candidate is not None:
        candidate_sql, explanation = llm_candidate
        try:
            safe_sql = _validate_sql(candidate_sql)
            rows = _execute_safe_sql(safe_sql)
            return {
                'mode': 'llm_text2sql',
                'sql': safe_sql,
                'params': (),
                'rows': rows,
                'explanation': explanation,
            }
        except Exception as exc:
            llm_error = str(exc)

    local_sql, params, explanation = _build_local_sql(question)
    safe_sql = _validate_sql(local_sql)
    rows = _execute_safe_sql(safe_sql, params)
    result = {
        'mode': 'heuristic_text2sql',
        'sql': safe_sql,
        'params': params,
        'rows': rows,
        'explanation': explanation,
    }
    if llm_error:
        result['llm_fallback_error'] = llm_error
    return result


def get_market_summary_for_role(target_role: str) -> dict[str, Any]:
    init_sqlite()
    like_role = f'%{target_role.lower()}%'
    with get_connection() as conn:
        count = conn.execute('SELECT COUNT(*) AS total FROM jobs WHERE LOWER(job_title) LIKE ?', (like_role,)).fetchone()['total']
        locations = [row['location'] for row in conn.execute(
            'SELECT location, COUNT(*) AS total FROM jobs WHERE LOWER(job_title) LIKE ? AND location IS NOT NULL GROUP BY location ORDER BY total DESC LIMIT 5',
            (like_role,),
        )]
        sample_titles = [row['job_title'] for row in conn.execute(
            'SELECT job_title FROM jobs WHERE LOWER(job_title) LIKE ? ORDER BY scrape_timestamp DESC LIMIT 5',
            (like_role,),
        )]
    return {'matching_jobs': count, 'top_locations': locations, 'sample_titles': sample_titles}


def get_embeddings() -> Any:
    try:
        from langchain_openai import OpenAIEmbeddings
    except Exception as exc:
        raise RuntimeError(f'langchain_openai belum tersedia: {exc}') from exc
    settings = get_settings()
    return OpenAIEmbeddings(model=settings.embedding_model, openai_api_key=settings.openai_api_key)


def get_qdrant_client() -> Any:
    try:
        from qdrant_client import QdrantClient
    except Exception as exc:
        raise RuntimeError(f'qdrant_client belum tersedia: {exc}') from exc
    settings = get_settings()
    if not settings.qdrant_url:
        raise RuntimeError('QDRANT_URL belum diisi')
    return QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key or None)


def ensure_collection(vector_size: int = 1536) -> None:
    try:
        from qdrant_client.models import Distance, VectorParams
    except Exception as exc:
        raise RuntimeError(f'qdrant_client belum tersedia: {exc}') from exc
    settings = get_settings()
    client = get_qdrant_client()
    collections = {c.name for c in client.get_collections().collections}
    if settings.qdrant_collection_name not in collections:
        client.create_collection(
            collection_name=settings.qdrant_collection_name,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )


def get_vector_store() -> Any:
    try:
        from langchain_qdrant import QdrantVectorStore
    except Exception as exc:
        raise RuntimeError(f'langchain_qdrant belum tersedia: {exc}') from exc
    settings = get_settings()
    ensure_collection()
    return QdrantVectorStore(
        client=get_qdrant_client(),
        collection_name=settings.qdrant_collection_name,
        embedding=get_embeddings(),
    )
