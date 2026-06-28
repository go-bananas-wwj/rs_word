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


def test_review_page():
    response = client.get("/review")
    assert response.status_code == 200
    assert "河流笔画筛选" in response.text


def test_swipe_page():
    response = client.get("/swipe")
    assert response.status_code == 200
    assert "河流笔画卡片筛选" in response.text


def test_save_review_selection(monkeypatch, tmp_path):
    selections_path = tmp_path / "review_selections.json"
    monkeypatch.setattr(
        "rs_words.web._load_review_rows",
        lambda: [{"chip_id": "heng_001", "stroke_type": "heng"}],
    )
    monkeypatch.setattr("rs_words.web.REVIEW_SELECTIONS_PATH", selections_path)

    response = client.post(
        "/api/review/selection",
        json={"chip_id": "heng_001", "decision": "selected", "note": "good"},
    )

    assert response.status_code == 200
    assert '"decision": "selected"' in selections_path.read_text(encoding="utf-8")
