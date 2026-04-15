from fastapi.testclient import TestClient

from karierai.server import app


def test_chat_fallback() -> None:
    client = TestClient(app)
    response = client.post('/chat', json={'query': 'Cari lowongan data analyst di Jakarta', 'history': ''})
    assert response.status_code == 200
    payload = response.json()
    assert 'response' in payload
    assert 'used_tools' in payload


def test_chat_sql_path() -> None:
    client = TestClient(app)
    response = client.post('/chat', json={'query': 'Berapa rata-rata gaji data analyst per lokasi?', 'history': ''})
    assert response.status_code == 200
    payload = response.json()
    assert any(tool in payload['used_tools'] for tool in ['sql_query_jobs', 'route_task'])
