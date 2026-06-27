from fastapi.testclient import TestClient

from rs_words.web import app

client = TestClient(app)


def test_index():
    response = client.get("/")
    assert response.status_code == 200
    assert "河流汉字" in response.text


def test_create_missing_text():
    response = client.post("/api/create", data={})
    assert response.status_code == 422
