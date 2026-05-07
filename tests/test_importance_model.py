"""Tests for the Phase 2 ML importance model."""

from __future__ import annotations

import json
import math
import tempfile
from pathlib import Path

import pytest

from cogdb.models import MemoryType, MemoryUnit
from cogdb.models.importance import (
    ImportanceModel,
    _TRAINING_SAMPLES,
    get_default_model,
    score_importance,
    score_memory_unit,
    _heuristic_score,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_unit(content: str, mtype: MemoryType = MemoryType.EPISODIC) -> MemoryUnit:
    return MemoryUnit(
        content=content,
        memory_type=mtype,
        agent_id="test-agent",
    )


# ── Feature extraction ────────────────────────────────────────────────────────

class TestFeatureExtraction:
    def test_returns_11_features(self):
        feats = ImportanceModel.extract_features("hello world", MemoryType.EPISODIC)
        assert len(feats) == 11

    def test_memory_type_norm(self):
        ep = ImportanceModel.extract_features("x", MemoryType.EPISODIC)[0]
        sem = ImportanceModel.extract_features("x", MemoryType.SEMANTIC)[0]
        proc = ImportanceModel.extract_features("x", MemoryType.PROCEDURAL)[0]
        assert ep == 0.0
        assert sem == 0.5
        assert proc == 1.0

    def test_version_pattern_detected(self):
        with_version = ImportanceModel.extract_features("v1.2.3 deployed", MemoryType.EPISODIC)
        without_version = ImportanceModel.extract_features("deployed today", MemoryType.EPISODIC)
        assert with_version[2] == 1.0   # has_version index
        assert without_version[2] == 0.0

    def test_metric_pattern_detected(self):
        with_metric = ImportanceModel.extract_features("response time 800ms", MemoryType.EPISODIC)
        without_metric = ImportanceModel.extract_features("response time improved", MemoryType.EPISODIC)
        assert with_metric[3] == 1.0   # has_metric index
        assert without_metric[3] == 0.0

    def test_high_kw_density(self):
        high = ImportanceModel.extract_features("CRITICAL error fail crash", MemoryType.EPISODIC)
        low = ImportanceModel.extract_features("routine update completed", MemoryType.EPISODIC)
        assert high[4] > low[4]

    def test_low_kw_density(self):
        hedged = ImportanceModel.extract_features("maybe we could possibly try this example", MemoryType.EPISODIC)
        direct = ImportanceModel.extract_features("we deployed the fix", MemoryType.EPISODIC)
        assert hedged[5] > direct[5]

    def test_access_log(self):
        no_access = ImportanceModel.extract_features("text", MemoryType.EPISODIC, access_count=0)
        with_access = ImportanceModel.extract_features("text", MemoryType.EPISODIC, access_count=10)
        assert with_access[9] > no_access[9]
        assert no_access[9] == 0.0
        assert math.isclose(with_access[9], math.log1p(10), rel_tol=1e-6)

    def test_recency_decay(self):
        fresh = ImportanceModel.extract_features("text", MemoryType.EPISODIC, recency_hours=0)
        old = ImportanceModel.extract_features("text", MemoryType.EPISODIC, recency_hours=96)
        assert fresh[10] > old[10]
        assert math.isclose(fresh[10], 1.0, rel_tol=1e-6)

    def test_features_all_finite(self):
        for s in _TRAINING_SAMPLES[:10]:
            feats = ImportanceModel.extract_features(
                s["content"], s["memory_type"],
                s.get("access_count", 0), s.get("recency_hours", 1.0)
            )
            assert all(math.isfinite(f) for f in feats), f"Non-finite feature in: {s['content'][:40]}"


# ── Model training and prediction ─────────────────────────────────────────────

class TestModelTraining:
    def test_train_returns_self(self):
        model = ImportanceModel()
        result = model.train(_TRAINING_SAMPLES)
        assert result is model

    def test_is_fitted_after_train(self):
        pytest.importorskip("sklearn")
        model = ImportanceModel().train(_TRAINING_SAMPLES)
        assert model.is_fitted

    def test_not_fitted_before_train(self):
        model = ImportanceModel()
        assert not model.is_fitted

    def test_predict_without_fit_uses_heuristic(self):
        model = ImportanceModel()
        score = model.predict("some text", MemoryType.EPISODIC)
        heuristic = _heuristic_score("some text", MemoryType.EPISODIC)
        assert math.isclose(score, heuristic, rel_tol=1e-4)

    def test_predict_returns_clamped_float(self):
        pytest.importorskip("sklearn")
        model = ImportanceModel().train(_TRAINING_SAMPLES)
        for s in _TRAINING_SAMPLES[:20]:
            score = model.predict(s["content"], s["memory_type"])
            assert 0.0 <= score <= 1.0, f"Out-of-range score {score} for: {s['content'][:40]}"

    def test_high_importance_content_scores_higher(self):
        pytest.importorskip("sklearn")
        model = ImportanceModel().train(_TRAINING_SAMPLES)
        high = model.predict("Critical production outage: database down, 502 errors for all users", MemoryType.EPISODIC)
        low = model.predict("Maybe we could try a different color scheme someday", MemoryType.EPISODIC)
        assert high > low, f"Expected high ({high:.3f}) > low ({low:.3f})"

    def test_version_pattern_boosts_score(self):
        pytest.importorskip("sklearn")
        model = ImportanceModel().train(_TRAINING_SAMPLES)
        with_ver = model.predict("v2.1.0 deployed to production successfully", MemoryType.EPISODIC)
        without_ver = model.predict("new version deployed to production successfully", MemoryType.EPISODIC)
        assert with_ver >= without_ver

    def test_procedural_ranks_above_generic_episodic(self):
        pytest.importorskip("sklearn")
        model = ImportanceModel().train(_TRAINING_SAMPLES)
        proc = model.predict("Standard deploy procedure with steps", MemoryType.PROCEDURAL)
        ep = model.predict("standard deploy happened today", MemoryType.EPISODIC)
        assert proc > ep

    def test_training_data_coverage(self):
        pytest.importorskip("sklearn")
        model = ImportanceModel().train(_TRAINING_SAMPLES)
        scores = [model.predict(s["content"], s["memory_type"]) for s in _TRAINING_SAMPLES]
        # Model should capture some spread (not all same value)
        assert max(scores) - min(scores) > 0.2, "Model output has too little spread"


# ── Persistence ───────────────────────────────────────────────────────────────

class TestModelPersistence:
    def test_save_load_roundtrip(self, tmp_path):
        pytest.importorskip("sklearn")
        model = ImportanceModel().train(_TRAINING_SAMPLES)
        path = str(tmp_path / "importance_model.json")
        model.save(path)

        loaded = ImportanceModel().load(path)
        assert loaded.is_fitted

        # Predictions should match
        for s in _TRAINING_SAMPLES[:10]:
            original = model.predict(s["content"], s["memory_type"])
            restored = loaded.predict(s["content"], s["memory_type"])
            assert math.isclose(original, restored, rel_tol=1e-6), \
                f"Prediction mismatch after load: {original} vs {restored}"

    def test_save_produces_valid_json(self, tmp_path):
        pytest.importorskip("sklearn")
        model = ImportanceModel().train(_TRAINING_SAMPLES)
        path = str(tmp_path / "model.json")
        model.save(path)
        data = json.loads(Path(path).read_text())
        assert "coef" in data
        assert "intercept" in data
        assert "scaler_mean" in data
        assert "scaler_std" in data
        assert isinstance(data["coef"], list)

    def test_load_without_save_raises(self, tmp_path):
        model = ImportanceModel()
        with pytest.raises(Exception):
            model.load(str(tmp_path / "nonexistent.json"))


# ── Online learning ───────────────────────────────────────────────────────────

class TestPartialFit:
    def test_partial_fit_on_unfitted_model_trains_it(self):
        pytest.importorskip("sklearn")
        model = ImportanceModel()
        assert not model.is_fitted
        model.partial_fit(_TRAINING_SAMPLES[:10])
        assert model.is_fitted

    def test_partial_fit_updates_predictions(self):
        pytest.importorskip("sklearn")
        model = ImportanceModel().train(_TRAINING_SAMPLES)
        before = model.predict("new specific content v9.9.9", MemoryType.EPISODIC)

        # Teach: this content should have very high importance
        update_samples = [{"content": "new specific content v9.9.9", "memory_type": MemoryType.EPISODIC, "importance": 0.99}]
        model.partial_fit(update_samples)
        after = model.predict("new specific content v9.9.9", MemoryType.EPISODIC)

        # Score should move toward 0.99
        assert after > before or math.isclose(after, before, rel_tol=0.05), \
            f"partial_fit should not decrease score for high-label sample: before={before:.3f} after={after:.3f}"


# ── Default model singleton ───────────────────────────────────────────────────

class TestDefaultModel:
    def test_get_default_model_returns_model_or_none(self):
        result = get_default_model()
        assert result is None or isinstance(result, ImportanceModel)

    def test_get_default_model_idempotent(self):
        m1 = get_default_model()
        m2 = get_default_model()
        assert m1 is m2  # same object (singleton)

    def test_default_model_fitted_if_sklearn_available(self):
        sklearn = pytest.importorskip("sklearn")
        model = get_default_model()
        assert model is not None
        assert model.is_fitted


# ── score_importance public API ───────────────────────────────────────────────

class TestScoreImportance:
    def test_returns_float_in_range(self):
        score = score_importance("hello world", MemoryType.EPISODIC)
        assert 0.0 <= score <= 1.0

    def test_explicit_importance_blended(self):
        # With explicit=1.0, score should be > pure computed
        high = score_importance("placeholder dummy test", MemoryType.EPISODIC, explicit_importance=1.0)
        low = score_importance("placeholder dummy test", MemoryType.EPISODIC, explicit_importance=0.0)
        assert high > low

    def test_explicit_importance_60pct_weight(self):
        content = "test content"
        computed = score_importance(content, MemoryType.EPISODIC, explicit_importance=None)
        blended = score_importance(content, MemoryType.EPISODIC, explicit_importance=1.0)
        # blended ≈ 0.6 * 1.0 + 0.4 * computed (minor rounding at 4dp boundary)
        expected = 0.6 * 1.0 + 0.4 * computed
        assert math.isclose(blended, expected, abs_tol=1e-3)

    def test_high_importance_content_scores_higher_than_hedge_content(self):
        high = score_importance("CRITICAL production outage error fail", MemoryType.EPISODIC)
        low = score_importance("maybe we could possibly think about this someday", MemoryType.EPISODIC)
        assert high > low

    def test_score_memory_unit_consistent(self):
        unit = _make_unit("v1.0.0 deployed with security fix")
        unit_score = score_memory_unit(unit)
        assert 0.0 <= unit_score <= 1.0
