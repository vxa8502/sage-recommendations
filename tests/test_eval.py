"""Tests for sage.data.eval — evaluation dataset loading utilities."""

import json

import pytest

from sage.data.eval import load_eval_cases


class TestLoadEvalCases:
    """Tests for load_eval_cases function."""

    def test_valid_cases(self, tmp_path, monkeypatch):
        """Valid JSON file returns list of EvalCase objects."""
        monkeypatch.setattr("sage.data.eval.EVAL_DIR", tmp_path)

        data = [
            {
                "query": "wireless headphones",
                "relevant_items": {"B001": 3.0, "B002": 2.0},
            },
            {"query": "bluetooth speaker", "relevant_items": {"B003": 1.0}},
        ]
        (tmp_path / "test.json").write_text(json.dumps(data))

        cases = load_eval_cases("test.json")

        assert len(cases) == 2
        assert cases[0].query == "wireless headphones"
        assert cases[0].relevant_items == {"B001": 3.0, "B002": 2.0}
        assert cases[1].query == "bluetooth speaker"

    def test_empty_list_returns_empty(self, tmp_path, monkeypatch):
        """Empty JSON array returns empty list without error."""
        monkeypatch.setattr("sage.data.eval.EVAL_DIR", tmp_path)

        (tmp_path / "empty.json").write_text("[]")

        cases = load_eval_cases("empty.json")

        assert cases == []

    def test_file_not_found_raises_clear_error(self, tmp_path, monkeypatch):
        """Missing file raises FileNotFoundError with filepath context."""
        monkeypatch.setattr("sage.data.eval.EVAL_DIR", tmp_path)

        with pytest.raises(FileNotFoundError, match="Evaluation file not found"):
            load_eval_cases("nonexistent.json")

    def test_invalid_json_raises_clear_error(self, tmp_path, monkeypatch):
        """Invalid JSON raises ValueError with line/column info."""
        monkeypatch.setattr("sage.data.eval.EVAL_DIR", tmp_path)

        (tmp_path / "bad.json").write_text("{invalid json")

        with pytest.raises(ValueError, match="Invalid JSON format"):
            load_eval_cases("bad.json")

    def test_not_array_raises_error(self, tmp_path, monkeypatch):
        """JSON object (not array) raises ValueError."""
        monkeypatch.setattr("sage.data.eval.EVAL_DIR", tmp_path)

        (tmp_path / "object.json").write_text('{"query": "test"}')

        with pytest.raises(ValueError, match="must contain a JSON array"):
            load_eval_cases("object.json")

    def test_missing_query_first_case(self, tmp_path, monkeypatch):
        """Missing query in first case raises ValueError with index."""
        monkeypatch.setattr("sage.data.eval.EVAL_DIR", tmp_path)

        data = [{"relevant_items": {"B001": 1.0}}]
        (tmp_path / "test.json").write_text(json.dumps(data))

        with pytest.raises(ValueError, match="Missing 'query' field in case 0"):
            load_eval_cases("test.json")

    def test_missing_query_later_case(self, tmp_path, monkeypatch):
        """Missing query in later case raises ValueError with correct index."""
        monkeypatch.setattr("sage.data.eval.EVAL_DIR", tmp_path)

        data = [
            {"query": "valid", "relevant_items": {"B001": 1.0}},
            {"query": "also valid", "relevant_items": {"B002": 2.0}},
            {"relevant_items": {"B003": 3.0}},  # Missing query at index 2
        ]
        (tmp_path / "test.json").write_text(json.dumps(data))

        with pytest.raises(ValueError, match="Missing 'query' field in case 2"):
            load_eval_cases("test.json")

    def test_missing_relevant_items(self, tmp_path, monkeypatch):
        """Missing relevant_items raises ValueError with index."""
        monkeypatch.setattr("sage.data.eval.EVAL_DIR", tmp_path)

        data = [{"query": "test query"}]
        (tmp_path / "test.json").write_text(json.dumps(data))

        with pytest.raises(
            ValueError, match="Missing 'relevant_items' field in case 0"
        ):
            load_eval_cases("test.json")

    def test_relevant_items_not_dict(self, tmp_path, monkeypatch):
        """relevant_items as list raises ValueError."""
        monkeypatch.setattr("sage.data.eval.EVAL_DIR", tmp_path)

        data = [{"query": "test", "relevant_items": ["B001", "B002"]}]
        (tmp_path / "test.json").write_text(json.dumps(data))

        with pytest.raises(ValueError, match="'relevant_items' must be a dict"):
            load_eval_cases("test.json")

    def test_relevance_score_not_numeric(self, tmp_path, monkeypatch):
        """Non-numeric relevance score raises ValueError with product ID."""
        monkeypatch.setattr("sage.data.eval.EVAL_DIR", tmp_path)

        data = [{"query": "test", "relevant_items": {"B001": "high"}}]
        (tmp_path / "test.json").write_text(json.dumps(data))

        with pytest.raises(
            ValueError, match="Relevance score for 'B001' must be numeric"
        ):
            load_eval_cases("test.json")

    def test_relevance_score_as_int_accepted(self, tmp_path, monkeypatch):
        """Integer relevance scores are accepted."""
        monkeypatch.setattr("sage.data.eval.EVAL_DIR", tmp_path)

        data = [{"query": "test", "relevant_items": {"B001": 3}}]
        (tmp_path / "test.json").write_text(json.dumps(data))

        cases = load_eval_cases("test.json")

        assert cases[0].relevant_items["B001"] == 3

    def test_user_id_optional(self, tmp_path, monkeypatch):
        """user_id field is optional."""
        monkeypatch.setattr("sage.data.eval.EVAL_DIR", tmp_path)

        data = [{"query": "test", "relevant_items": {"B001": 1.0}}]
        (tmp_path / "test.json").write_text(json.dumps(data))

        cases = load_eval_cases("test.json")

        assert cases[0].user_id is None

    def test_user_id_preserved(self, tmp_path, monkeypatch):
        """user_id field is preserved when present."""
        monkeypatch.setattr("sage.data.eval.EVAL_DIR", tmp_path)

        data = [{"query": "test", "relevant_items": {"B001": 1.0}, "user_id": "U123"}]
        (tmp_path / "test.json").write_text(json.dumps(data))

        cases = load_eval_cases("test.json")

        assert cases[0].user_id == "U123"

    def test_extra_fields_ignored(self, tmp_path, monkeypatch):
        """Extra fields (category, intent) are ignored without error."""
        monkeypatch.setattr("sage.data.eval.EVAL_DIR", tmp_path)

        data = [
            {
                "query": "smart speaker",
                "relevant_items": {"B001": 3.0},
                "category": "echo_devices",
                "intent": "feature_specific",
            }
        ]
        (tmp_path / "test.json").write_text(json.dumps(data))

        cases = load_eval_cases("test.json")

        assert len(cases) == 1
        assert cases[0].query == "smart speaker"

    def test_relevant_set_works_after_load(self, tmp_path, monkeypatch):
        """Loaded cases have working relevant_set property."""
        monkeypatch.setattr("sage.data.eval.EVAL_DIR", tmp_path)

        data = [
            {"query": "test", "relevant_items": {"B001": 3.0, "B002": 0.0, "B003": 1.0}}
        ]
        (tmp_path / "test.json").write_text(json.dumps(data))

        cases = load_eval_cases("test.json")

        # relevant_set should only include items with score > 0
        assert cases[0].relevant_set == {"B001", "B003"}
