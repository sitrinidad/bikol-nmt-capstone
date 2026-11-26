"""API rules from the Phase 3 spec. A recording engine stands in for the 2.5 GB model,
so these tests check the middleware (validation, JSON contract, database), not translation quality."""
import os
from pathlib import Path

import pymysql
import pytest
from fastapi.testclient import TestClient

from backend.config import Settings
from backend.main import create_app
from backend.nmt import EngineState, ModelNotReady, NMTEngine, TranslationResult
from database import FeedbackRepository, MySQLConfig

# Feedback tests use a separate database (dropped and recreated per test) on the same MySQL server as the app.
_base = Settings()
TEST_DB = MySQLConfig(
    host=os.getenv("TEST_MYSQL_HOST", _base.mysql_host),
    port=int(os.getenv("TEST_MYSQL_PORT", str(_base.mysql_port))),
    user=os.getenv("TEST_MYSQL_USER", _base.mysql_user),
    password=os.getenv("TEST_MYSQL_PASSWORD", _base.mysql_password),
    database=os.getenv("TEST_MYSQL_DATABASE", "bikole_test"),
)


def _server_conn(database=None):
    return pymysql.connect(host=TEST_DB.host, port=TEST_DB.port, user=TEST_DB.user, password=TEST_DB.password,
                           database=database, charset="utf8mb4", connect_timeout=3)


def _mysql_up():
    try:
        _server_conn().close()
        return True
    except pymysql.MySQLError:
        return False


needs_mysql = pytest.mark.skipif(not _mysql_up(), reason=f"MySQL not reachable at {TEST_DB.host}:{TEST_DB.port}")


def query(sql, args=()):
    conn = _server_conn(TEST_DB.database)
    try:
        with conn.cursor() as cur:
            cur.execute(sql, args)
            return cur.fetchall()
    finally:
        conn.close()


class RecordingEngine(NMTEngine):
    def __init__(self, state=EngineState.READY):
        super().__init__(Path("unused"), num_beams=4)
        self.state = state
        self.model_version = "test-adapter"
        self.calls = []

    def translate(self, text, source_lang, target_lang):
        if not self.ready:
            raise ModelNotReady("The translation model is still loading. Please try again in a moment.")
        self.calls.append((text, source_lang, target_lang))
        return TranslationResult(translated_text=f"<{target_lang}> {text}", confidence_score=0.8731, inference_latency_ms=142)


@pytest.fixture
def make_client(tmp_path):
    if _mysql_up():
        conn = _server_conn()
        with conn.cursor() as cur:
            cur.execute(f"DROP DATABASE IF EXISTS `{TEST_DB.database}`")
        conn.close()

    def _make(engine=None, db=TEST_DB):
        engine = engine or RecordingEngine()
        settings = Settings(load_model=False, frontend_dir=tmp_path / "no-frontend")
        app = create_app(settings=settings, engine=engine, repo=FeedbackRepository(db))
        return TestClient(app), engine, db
    return _make


def test_translate_returns_the_four_spec_fields_from_the_engine(make_client):
    client, engine, _ = make_client()
    with client:
        r = client.post("/api/v1/translate", json={"source_text": "Where are you going?", "source_lang": "eng", "target_lang": "bcl_North"})
    assert r.status_code == 200
    assert r.json() == {"status": "success", "translated_text": "<bcl_North> Where are you going?",
                        "confidence_score": 0.8731, "inference_latency_ms": 142}
    assert engine.calls == [("Where are you going?", "eng", "bcl_North")]


@pytest.mark.parametrize("src,tgt", [("eng", "bcl_South"), ("bcl_South", "eng"), ("eng", "bcl_North"),
                                     ("bcl_North", "eng"), ("bcl_South", "bcl_North"), ("bcl_North", "bcl_South")])
def test_all_six_trained_directions_are_accepted(make_client, src, tgt):
    client, _, _ = make_client()
    with client:
        r = client.post("/api/v1/translate", json={"source_text": "Hello", "source_lang": src, "target_lang": tgt})
    assert r.status_code == 200


def test_same_source_and_target_language_is_blocked(make_client):
    client, engine, _ = make_client()
    with client:
        r = client.post("/api/v1/translate", json={"source_text": "Hello", "source_lang": "eng", "target_lang": "eng"})
    assert r.status_code == 422
    assert r.json() == {"status": "failed", "error": "Source and target language must be different."}
    assert engine.calls == []


def test_direction_the_model_was_not_trained_on_is_rejected(make_client):
    engine = RecordingEngine()
    engine.directions = [("eng", "bcl_North"), ("bcl_North", "eng")]   # e.g. the "ablation" adapter
    client, _, _ = make_client(engine)
    with client:
        r = client.post("/api/v1/translate", json={"source_text": "Hello", "source_lang": "eng", "target_lang": "bcl_South"})
    assert r.status_code == 400
    assert r.json()["status"] == "failed"
    assert engine.calls == []


def test_unknown_language_code_is_rejected(make_client):
    client, _, _ = make_client()
    with client:
        r = client.post("/api/v1/translate", json={"source_text": "Hello", "source_lang": "tgl_Latn", "target_lang": "eng"})
    assert r.status_code == 422
    assert r.json()["status"] == "failed"


def test_blank_source_text_is_rejected(make_client):
    client, engine, _ = make_client()
    with client:
        r = client.post("/api/v1/translate", json={"source_text": "   ", "source_lang": "eng", "target_lang": "bcl_South"})
    assert r.status_code == 422
    assert r.json()["error"] == "Source text must not be empty."
    assert engine.calls == []


def test_source_text_over_1000_characters_is_rejected(make_client):
    client, _, _ = make_client()
    with client:
        r = client.post("/api/v1/translate", json={"source_text": "a" * 1001, "source_lang": "eng", "target_lang": "bcl_South"})
    assert r.status_code == 422


def test_translate_reports_failure_instead_of_fake_output_while_model_loads(make_client):
    client, _, _ = make_client(RecordingEngine(state=EngineState.LOADING))
    with client:
        r = client.post("/api/v1/translate", json={"source_text": "Hello", "source_lang": "eng", "target_lang": "bcl_South"})
    assert r.status_code == 503
    body = r.json()
    assert body["status"] == "failed"
    assert "translated_text" not in body


def test_languages_endpoint_lists_three_languages_and_model_directions(make_client):
    client, _, _ = make_client()
    with client:
        body = client.get("/api/v1/languages").json()
    assert [l["code"] for l in body["languages"]] == ["eng", "bcl_South", "bcl_North"]
    assert len(body["directions"]) == 6


VALID_FEEDBACK = {
    "source_text": "Dai mo na sya paghabulon.",
    "source_lang": "bcl_South",
    "target_lang": "eng",
    "generated_translation": "Don't chase him.",
    "corrected_translation": "Don't chase after him/her anymore.",
    "explanation": "Missing 'anymore'.",
}


@needs_mysql
def test_feedback_is_saved_to_mysql(make_client):
    client, _, db = make_client()
    with client:
        r = client.post("/api/v1/feedback", json=VALID_FEEDBACK)
    assert r.status_code == 201
    assert r.json()["status"] == "success"
    row = query(
        "SELECT source_text, source_lang, target_lang, generated_translation, corrected_translation, explanation, model_version "
        "FROM feedback WHERE id = %s", (r.json()["feedback_id"],))[0]
    assert row == ("Dai mo na sya paghabulon.", "bcl_South", "eng", "Don't chase him.",
                   "Don't chase after him/her anymore.", "Missing 'anymore'.", "test-adapter")


@needs_mysql
def test_feedback_explanation_is_optional(make_client):
    client, _, db = make_client()
    payload = {k: v for k, v in VALID_FEEDBACK.items() if k != "explanation"}
    with client:
        r = client.post("/api/v1/feedback", json=payload)
    assert r.status_code == 201
    assert query("SELECT explanation FROM feedback") == ((None,),)


@needs_mysql
@pytest.mark.parametrize("field", ["source_text", "generated_translation", "corrected_translation"])
def test_feedback_requires_non_empty_fields(make_client, field):
    client, _, db = make_client()
    with client:
        r = client.post("/api/v1/feedback", json={**VALID_FEEDBACK, field: "  "})
    assert r.status_code == 422
    assert r.json()["status"] == "failed"
    assert query("SELECT COUNT(*) FROM feedback") == ((0,),)


@needs_mysql
def test_feedback_with_same_languages_is_rejected(make_client):
    client, _, db = make_client()
    with client:
        r = client.post("/api/v1/feedback", json={**VALID_FEEDBACK, "target_lang": "bcl_South"})
    assert r.status_code == 422
    assert query("SELECT COUNT(*) FROM feedback") == ((0,),)


def test_feedback_reports_failure_when_mysql_is_unreachable(make_client):
    down = MySQLConfig(host="127.0.0.1", port=1, user="root", password="", database="bikole_test")
    client, _, _ = make_client(db=down)
    with client:
        r = client.post("/api/v1/feedback", json=VALID_FEEDBACK)
        health = client.get("/api/v1/health").json()
    assert r.status_code == 503
    assert r.json()["status"] == "failed"
    assert "MySQL" in r.json()["error"]
    assert health["database"]["state"] == "unavailable"
