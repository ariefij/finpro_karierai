from __future__ import annotations

import mimetypes
import os
import time
from typing import Any, Iterator

import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()
API_URL = os.getenv('API_URL', 'http://localhost:8080')


def call_api(path: str, payload: dict[str, Any]) -> dict[str, Any]:
    response = requests.post(f'{API_URL}{path}', json=payload, timeout=120)
    response.raise_for_status()
    return response.json()


def call_api_file(path: str, file_name: str, file_bytes: bytes, extra_data: dict[str, Any] | None = None) -> dict[str, Any]:
    mime_type = mimetypes.guess_type(file_name)[0] or 'application/octet-stream'
    files = {'file': (file_name, file_bytes, mime_type)}
    response = requests.post(f'{API_URL}{path}', files=files, data=extra_data or {}, timeout=120)
    response.raise_for_status()
    return response.json()


def stream_text_chunks(text: str) -> Iterator[str]:
    words = text.split()
    for index, word in enumerate(words):
        suffix = ' ' if index < len(words) - 1 else ''
        yield word + suffix
        time.sleep(0.02)


def render_usage_badges(message: dict[str, Any]) -> None:
    if message.get('role') != 'assistant':
        return
    input_tokens = int(message.get('input_tokens', 0) or 0)
    output_tokens = int(message.get('output_tokens', 0) or 0)
    total_tokens = int(message.get('total_tokens', input_tokens + output_tokens) or 0)
    token_mode = message.get('token_mode', 'estimated')
    if not any([input_tokens, output_tokens, total_tokens]):
        return
    label = 'provider' if token_mode == 'provider_usage' else 'estimasi'
    st.caption(f'Input tokens: {input_tokens} | Output tokens: {output_tokens} | Total: {total_tokens} ({label})')


st.set_page_config(page_title='KarierAI', layout='wide')
st.title('KarierAI')

with st.sidebar:
    st.markdown(f'**API URL**: `{API_URL}`')
    st.caption('CV upload mendukung PDF teks, PDF scan, dan gambar (PNG/JPG/JPEG/WEBP/BMP/TIFF).')
    debug_mode = st.checkbox('Debug mode', value=False)
    show_token_usage = st.checkbox('Tampilkan token usage', value=True)
    ui_streaming = st.checkbox('Streaming jawaban di UI', value=True)
    if st.button('Reset chat'):
        st.session_state.messages = []

chat_tab, cv_tab, consult_tab = st.tabs(['Chat', 'CV Analyzer', 'Career Consultation'])

with chat_tab:
    if 'messages' not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message['role']):
            st.markdown(message['content'])
            if show_token_usage:
                render_usage_badges(message)

    prompt = st.chat_input('Tanya tentang lowongan, statistik, atau konsultasi karier')
    if prompt:
        history = [
            {'role': item['role'], 'content': item['content']}
            for item in st.session_state.messages[-10:]
            if item.get('content')
        ]
        user_message = {'role': 'user', 'content': prompt}
        st.session_state.messages.append(user_message)
        with st.chat_message('user'):
            st.markdown(prompt)
        with st.chat_message('assistant'):
            placeholder = st.empty()
            placeholder.markdown('Sedang menyusun jawaban...')
            try:
                result = call_api('/chat', {'query': prompt, 'history': history})
                assistant_text = result['response']
                if ui_streaming:
                    with placeholder.container():
                        st.write_stream(stream_text_chunks(assistant_text))
                else:
                    placeholder.markdown(assistant_text)
                assistant_message = {
                    'role': 'assistant',
                    'content': assistant_text,
                    'input_tokens': result.get('input_tokens', 0),
                    'output_tokens': result.get('output_tokens', 0),
                    'total_tokens': result.get('total_tokens', result.get('input_tokens', 0) + result.get('output_tokens', 0)),
                    'token_mode': result.get('token_mode', 'estimated'),
                    'used_tools': result.get('used_tools', []),
                }
                if show_token_usage:
                    render_usage_badges(assistant_message)
                if debug_mode:
                    with st.expander('Tool messages'):
                        st.code('\n\n'.join(result.get('tool_messages', [])) or 'No tool messages')
                    with st.expander('Usage'):
                        st.json(
                            {
                                'input_tokens': result.get('input_tokens', 0),
                                'output_tokens': result.get('output_tokens', 0),
                                'total_tokens': result.get('total_tokens', 0),
                                'token_mode': result.get('token_mode', 'estimated'),
                                'used_tools': result.get('used_tools', []),
                            }
                        )
                st.session_state.messages.append(assistant_message)
            except Exception as exc:
                placeholder.empty()
                st.error(f'Gagal memanggil API: {exc}')

with cv_tab:
    cv_text = st.text_area('Tempel teks CV di sini', height=260)
    cv_file = st.file_uploader('Atau upload CV PDF / gambar', type=['pdf', 'png', 'jpg', 'jpeg', 'webp', 'bmp', 'tif', 'tiff'])
    top_k = st.slider('Top K rekomendasi', min_value=1, max_value=10, value=5)
    col1, col2 = st.columns(2)
    if col1.button('Analisis CV'):
        try:
            if cv_file is not None:
                st.json(call_api_file('/cv/analyze-file', cv_file.name, cv_file.getvalue()))
            elif cv_text.strip():
                st.json(call_api('/cv/analyze', {'cv_text': cv_text}))
            else:
                st.warning('Isi teks CV atau upload PDF/gambar terlebih dahulu.')
        except Exception as exc:
            st.error(str(exc))
    if col2.button('Cari rekomendasi kerja'):
        try:
            if cv_file is not None:
                st.json(call_api_file('/recommend-file', cv_file.name, cv_file.getvalue(), {'top_k': top_k}))
            elif cv_text.strip():
                st.json(call_api('/recommend', {'cv_text': cv_text, 'top_k': top_k}))
            else:
                st.warning('Isi teks CV atau upload PDF/gambar terlebih dahulu.')
        except Exception as exc:
            st.error(str(exc))

with consult_tab:
    consult_cv = st.text_area('Teks CV untuk konsultasi karier', height=220, key='consult_cv')
    consult_file = st.file_uploader(
        'Atau upload CV PDF / gambar untuk konsultasi',
        type=['pdf', 'png', 'jpg', 'jpeg', 'webp', 'bmp', 'tif', 'tiff'],
        key='consult_pdf',
    )
    target_role = st.text_input('Target role', value='Data Analyst')
    if st.button('Analisis gap skill'):
        try:
            if consult_file is not None:
                st.json(call_api_file('/consult-file', consult_file.name, consult_file.getvalue(), {'target_role': target_role}))
            elif consult_cv.strip():
                st.json(call_api('/consult', {'cv_text': consult_cv, 'target_role': target_role}))
            else:
                st.warning('Isi teks CV atau upload PDF/gambar terlebih dahulu.')
        except Exception as exc:
            st.error(str(exc))
