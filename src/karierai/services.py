from __future__ import annotations

import re
import shutil
from collections import Counter
from io import BytesIO
from typing import Any

from .database import get_market_summary_for_role, search_jobs

SKILL_KEYWORDS = [
    'python', 'sql', 'excel', 'power bi', 'tableau', 'communication',
    'leadership', 'recruitment', 'payroll', 'analysis', 'machine learning',
    'statistics', 'dashboard', 'etl', 'data visualization', 'forecasting',
    'reporting', 'hris', 'talent acquisition', 'business intelligence',
    'r', 'spark', 'tensorflow', 'pytorch', 'project management',
]

ROLE_KEYWORDS = [
    'data analyst', 'business analyst', 'data scientist', 'machine learning engineer',
    'hr manager', 'recruiter', 'talent acquisition', 'payroll specialist',
    'business intelligence', 'product analyst', 'finance analyst',
]

EDUCATION_KEYWORDS = ['s1', 's2', 'sarjana', 'bachelor', 'master', 'phd']

ROLE_SKILLS = {
    'data analyst': ['sql', 'excel', 'tableau', 'power bi', 'analysis', 'statistics'],
    'data scientist': ['python', 'sql', 'machine learning', 'statistics'],
    'hr manager': ['leadership', 'recruitment', 'communication', 'payroll', 'hris'],
    'business analyst': ['sql', 'excel', 'dashboard', 'analysis', 'communication'],
}

_SUPPORTED_IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.webp', '.bmp', '.tif', '.tiff'}


def _get_ocr_languages() -> str:
    preferred = ['ind', 'eng']
    try:
        import pytesseract
    except Exception as exc:  # pragma: no cover - dependency guard
        raise RuntimeError(f'pytesseract belum terpasang: {exc}') from exc

    available: set[str] = set()
    try:
        result = pytesseract.get_languages(config='')
        available = {item.strip() for item in result if item.strip()}
    except Exception:
        pass

    selected = [lang for lang in preferred if lang in available]
    if not selected:
        selected = ['eng'] if 'eng' in available or not available else [next(iter(available))]
    return '+'.join(selected)


def _load_image(image_bytes: bytes):
    try:
        from PIL import Image, ImageOps
    except Exception as exc:  # pragma: no cover - dependency guard
        raise RuntimeError(f'Pillow belum terpasang: {exc}') from exc

    image = Image.open(BytesIO(image_bytes))
    image.load()
    return ImageOps.exif_transpose(image)


def _prepare_images_for_ocr(image) -> list[Any]:
    from PIL import ImageFilter, ImageOps

    variants = []
    base = image.convert('L')
    autocontrast = ImageOps.autocontrast(base)
    if min(autocontrast.size) < 1600:
        scale = max(2, int(1800 / max(1, min(autocontrast.size))))
        autocontrast = autocontrast.resize((autocontrast.width * scale, autocontrast.height * scale))
    variants.append(autocontrast)
    sharpened = autocontrast.filter(ImageFilter.SHARPEN)
    variants.append(sharpened)
    binary = sharpened.point(lambda px: 255 if px > 170 else 0)
    variants.append(binary)
    return variants


def _ocr_single_image(image) -> str:
    try:
        import pytesseract
    except Exception as exc:  # pragma: no cover - dependency guard
        raise RuntimeError(f'pytesseract belum terpasang: {exc}') from exc

    if not shutil.which('tesseract'):
        raise RuntimeError('Binary tesseract tidak ditemukan di PATH. Instal Tesseract OCR terlebih dahulu.')

    languages = _get_ocr_languages()
    candidates: list[str] = []
    configs = ['--psm 6', '--psm 3', '--psm 11']
    for prepared in _prepare_images_for_ocr(image):
        for config in configs:
            text = pytesseract.image_to_string(prepared, lang=languages, config=config) or ''
            cleaned = ' '.join(text.split())
            if cleaned:
                candidates.append(cleaned)
    best = max(candidates, key=len) if candidates else ''
    return _normalize_ocr_text(best)




def _normalize_ocr_text(text: str) -> str:
    normalized = ' '.join(text.split())
    replacements = {
        r'\bsol\b': 'sql',
        r'\b5ql\b': 'sql',
        r'\bsqi\b': 'sql',
        r'\bpyth0n\b': 'python',
        r'\btabieau\b': 'tableau',
        r'\bpowerbi\b': 'power bi',
    }
    for pattern, repl in replacements.items():
        normalized = re.sub(pattern, repl, normalized, flags=re.IGNORECASE)
    return normalized

def extract_text_from_image_bytes(image_bytes: bytes) -> str:
    image = _load_image(image_bytes)
    text = _ocr_single_image(image)
    if not text:
        raise ValueError('Gambar CV tidak berhasil dibaca. Pastikan teks terlihat jelas dan resolusinya cukup.')
    return text


def _render_pdf_to_images(pdf_bytes: bytes) -> list[Any]:
    try:
        import fitz
    except Exception as exc:  # pragma: no cover - dependency guard
        raise RuntimeError(f'PyMuPDF belum terpasang: {exc}') from exc

    try:
        from PIL import Image
    except Exception as exc:  # pragma: no cover - dependency guard
        raise RuntimeError(f'Pillow belum terpasang: {exc}') from exc

    document = fitz.open(stream=pdf_bytes, filetype='pdf')
    pages: list[Any] = []
    try:
        for page in document:
            pix = page.get_pixmap(matrix=fitz.Matrix(2.5, 2.5), alpha=False)
            pages.append(Image.open(BytesIO(pix.tobytes('png'))))
    finally:
        document.close()
    return pages


def _extract_text_pdf_native(pdf_bytes: bytes) -> str:
    try:
        from pypdf import PdfReader
    except Exception as exc:  # pragma: no cover - dependency guard
        raise RuntimeError(f'pypdf belum terpasang: {exc}') from exc

    reader = PdfReader(BytesIO(pdf_bytes))
    page_texts: list[str] = []
    for page in reader.pages:
        text = page.extract_text() or ''
        cleaned = ' '.join(text.split())
        if cleaned:
            page_texts.append(cleaned)
    return '\n'.join(page_texts).strip()


def _looks_like_useful_text(text: str) -> bool:
    compact = ' '.join(text.split())
    return len(compact) >= 20 and sum(ch.isalpha() for ch in compact) >= 10


def extract_text_from_pdf_bytes(pdf_bytes: bytes) -> str:
    """Extract text from a CV PDF.

    The function first tries native PDF text extraction. If the PDF is a scan/image-only
    document or the extracted text is too sparse, it falls back to OCR per page.
    """
    native_text = _extract_text_pdf_native(pdf_bytes)
    if _looks_like_useful_text(native_text):
        return native_text

    ocr_pages: list[str] = []
    for image in _render_pdf_to_images(pdf_bytes):
        text = _ocr_single_image(image)
        cleaned = ' '.join(text.split())
        if cleaned:
            ocr_pages.append(cleaned)

    combined_ocr = '\n'.join(ocr_pages).strip()
    if _looks_like_useful_text(combined_ocr):
        return combined_ocr
    if native_text:
        return native_text
    raise ValueError('PDF tidak berhasil dibaca. Pastikan file tidak rusak dan teks pada CV scan terlihat jelas.')


def extract_text_from_upload_bytes(file_name: str, content_type: str | None, raw_bytes: bytes) -> str:
    lowered_name = (file_name or '').lower()
    lowered_type = (content_type or '').lower()
    if lowered_name.endswith('.pdf') or lowered_type in {'application/pdf', 'application/x-pdf'}:
        return extract_text_from_pdf_bytes(raw_bytes)
    if any(lowered_name.endswith(ext) for ext in _SUPPORTED_IMAGE_EXTENSIONS) or lowered_type.startswith('image/'):
        return extract_text_from_image_bytes(raw_bytes)
    raise ValueError('Format file belum didukung. Upload CV dalam bentuk PDF, PNG, JPG, JPEG, WEBP, BMP, atau TIFF.')


def _find_keywords(text: str, keywords: list[str]) -> list[str]:
    lower = text.lower()
    found: list[str] = []
    for keyword in keywords:
        pattern = rf'(?<!\w){re.escape(keyword.lower())}(?!\w)'
        if re.search(pattern, lower):
            found.append(keyword)
    return sorted(set(found), key=lambda item: keywords.index(item))


def _extract_years(text: str) -> dict[str, int | list[int]]:
    lower = text.lower()
    matches = re.findall(r'(\d+)\+?\s*(?:tahun|years?)', lower)
    values = [int(match) for match in matches]
    return {'mentions': values, 'max_years': max(values) if values else 0}


def _extract_sentences(text: str, keywords: list[str], limit: int = 5) -> list[str]:
    sentences = re.split(r'(?<=[.!?\n])\s+', text.strip())
    selected: list[str] = []
    for sentence in sentences:
        lower = sentence.lower()
        if any(keyword in lower for keyword in keywords):
            cleaned = ' '.join(sentence.split())
            if cleaned and cleaned not in selected:
                selected.append(cleaned)
        if len(selected) >= limit:
            break
    return selected


def extract_cv_profile_data(cv_text: str) -> dict[str, object]:
    text = ' '.join(cv_text.split())
    skills = _find_keywords(text, SKILL_KEYWORDS)
    likely_roles = _find_keywords(text, ROLE_KEYWORDS)
    education = _find_keywords(text, EDUCATION_KEYWORDS)
    years = _extract_years(text)
    headline_candidates = _extract_sentences(text, ROLE_KEYWORDS + ['experience', 'berpengalaman'], limit=3)
    strengths = _extract_sentences(text, skills[:8] if skills else SKILL_KEYWORDS[:8], limit=6)
    return {
        'skills': skills,
        'likely_roles': likely_roles,
        'education_mentions': education,
        'years_of_experience_mentions': years['mentions'],
        'estimated_years_experience': years['max_years'],
        'headline': headline_candidates[0] if headline_candidates else text[:180],
        'strength_evidence': strengths,
        'text_excerpt': text[:1200],
    }


def summarize_skill_overlap(cv_skills: list[str], job_text: str, job_title: str = '') -> dict[str, object]:
    lower = f'{job_title} {job_text}'.lower()
    matched = [skill for skill in cv_skills if skill in lower]
    keyword_hits = Counter(matched)
    return {'matched_skills': sorted(keyword_hits), 'match_count': len(matched)}


def _score_job(profile: dict[str, object], job: dict[str, Any]) -> tuple[float, dict[str, object]]:
    cv_skills = profile.get('skills', []) if isinstance(profile.get('skills'), list) else []
    likely_roles = profile.get('likely_roles', []) if isinstance(profile.get('likely_roles'), list) else []
    years = int(profile.get('estimated_years_experience', 0) or 0)
    overlap = summarize_skill_overlap(cv_skills, str(job.get('job_description', '')), str(job.get('job_title', '')))
    matched_skills = overlap['matched_skills']
    match_count = int(overlap['match_count'])

    title = str(job.get('job_title', '')).lower()
    title_bonus = 0.0
    matched_roles: list[str] = []
    for role in likely_roles:
        if role in title:
            title_bonus += 2.5
            matched_roles.append(role)

    years_bonus = 0.0
    description = str(job.get('job_description', '')).lower()
    if years > 0 and f'{years}' in description:
        years_bonus = 0.5

    score = match_count * 1.5 + title_bonus + years_bonus
    explanation = []
    if matched_roles:
        explanation.append(f'Role CV selaras dengan judul lowongan: {", ".join(matched_roles)}')
    if matched_skills:
        explanation.append(f'Skill yang cocok: {", ".join(matched_skills[:6])}')
    if not explanation:
        explanation.append('Kecocokan dihitung dari kemiripan umum antara CV dan deskripsi lowongan.')

    return score, {
        'job_id': job.get('job_id'),
        'job_title': job.get('job_title'),
        'company_name': job.get('company_name'),
        'location': job.get('location'),
        'work_type': job.get('work_type'),
        'salary_raw': job.get('salary_raw'),
        'score': round(score, 2),
        'matched_skills': matched_skills,
        'explanation': explanation,
        'job_excerpt': str(job.get('job_description', ''))[:350],
    }


def build_recommendations(cv_text: str, top_k: int = 5) -> dict[str, object]:
    profile = extract_cv_profile_data(cv_text)
    search_query = ' '.join([*(profile.get('likely_roles') or []), *(profile.get('skills') or [])[:6]]).strip() or cv_text[:160]
    jobs = search_jobs(search_query=search_query, limit=max(top_k * 4, 10))
    scored = []
    for job in jobs:
        score, payload = _score_job(profile, job)
        if score > 0:
            scored.append((score, payload))
    scored.sort(key=lambda item: item[0], reverse=True)
    recommendations = [payload for _, payload in scored[:top_k]]
    return {'profile': profile, 'search_query': search_query, 'matches': recommendations}


def build_career_consultation(cv_text: str, target_role: str) -> dict[str, Any]:
    profile = extract_cv_profile_data(cv_text)
    role_key = target_role.lower().strip()
    required_skills = ROLE_SKILLS.get(role_key, [])
    cv_skills = set(profile.get('skills', []))
    matched = [skill for skill in required_skills if skill in cv_skills]
    missing = [skill for skill in required_skills if skill not in cv_skills]
    market = get_market_summary_for_role(target_role)

    recommendations: list[str] = []
    if missing:
        recommendations.append(f'Prioritaskan penguatan skill berikut: {", ".join(missing[:5])}.')
    if market['sample_titles']:
        recommendations.append(f'Pantau lowongan seperti: {", ".join(market["sample_titles"][:3])}.')
    if market['top_locations']:
        recommendations.append(f'Lokasi pasar kerja teratas untuk role ini: {", ".join(market["top_locations"][:3])}.')
    if not recommendations:
        recommendations.append('Profil sudah cukup dekat. Fokus pada portfolio, hasil kerja, dan kesiapan interview.')

    return {
        'target_role': target_role,
        'profile': profile,
        'matched_skills': matched,
        'missing_skills': missing,
        'market_summary': market,
        'recommendations': recommendations,
    }
