"""Tests for sage.data.loader — data loading and temporal split functions."""

import pandas as pd
import pytest

from sage.data.loader import (
    create_temporal_splits,
    filter_5_core,
    verify_temporal_boundaries,
)


class TestVerifyTemporalBoundaries:
    """Tests for verify_temporal_boundaries function."""

    def test_valid_splits_returns_boundaries(self):
        """Valid temporal splits should return boundary dict."""
        train_df = pd.DataFrame({"timestamp": [100, 200, 300]})
        val_df = pd.DataFrame({"timestamp": [400, 500]})
        test_df = pd.DataFrame({"timestamp": [600, 700, 800]})

        result = verify_temporal_boundaries(train_df, val_df, test_df, verbose=False)

        assert result == {
            "train": (100, 300),
            "val": (400, 500),
            "test": (600, 800),
        }

    def test_empty_train_raises_clear_error(self):
        """Empty train split should raise ValueError with clear message."""
        train_df = pd.DataFrame({"timestamp": []})
        val_df = pd.DataFrame({"timestamp": [100, 200]})
        test_df = pd.DataFrame({"timestamp": [300, 400]})

        with pytest.raises(ValueError, match="Train split is empty"):
            verify_temporal_boundaries(train_df, val_df, test_df, verbose=False)

    def test_empty_val_raises_clear_error(self):
        """Empty validation split should raise ValueError with clear message."""
        train_df = pd.DataFrame({"timestamp": [100, 200]})
        val_df = pd.DataFrame({"timestamp": []})
        test_df = pd.DataFrame({"timestamp": [300, 400]})

        with pytest.raises(ValueError, match="Validation split is empty"):
            verify_temporal_boundaries(train_df, val_df, test_df, verbose=False)

    def test_empty_test_raises_clear_error(self):
        """Empty test split should raise ValueError with clear message."""
        train_df = pd.DataFrame({"timestamp": [100, 200]})
        val_df = pd.DataFrame({"timestamp": [300, 400]})
        test_df = pd.DataFrame({"timestamp": []})

        with pytest.raises(ValueError, match="Test split is empty"):
            verify_temporal_boundaries(train_df, val_df, test_df, verbose=False)

    def test_missing_timestamp_column_train(self):
        """Missing timestamp column in train should raise ValueError."""
        train_df = pd.DataFrame({"other_col": [100, 200]})
        val_df = pd.DataFrame({"timestamp": [300, 400]})
        test_df = pd.DataFrame({"timestamp": [500, 600]})

        with pytest.raises(ValueError, match="Train split missing 'timestamp'"):
            verify_temporal_boundaries(train_df, val_df, test_df, verbose=False)

    def test_missing_timestamp_column_val(self):
        """Missing timestamp column in val should raise ValueError."""
        train_df = pd.DataFrame({"timestamp": [100, 200]})
        val_df = pd.DataFrame({"other_col": [300, 400]})
        test_df = pd.DataFrame({"timestamp": [500, 600]})

        with pytest.raises(ValueError, match="Validation split missing 'timestamp'"):
            verify_temporal_boundaries(train_df, val_df, test_df, verbose=False)

    def test_missing_timestamp_column_test(self):
        """Missing timestamp column in test should raise ValueError."""
        train_df = pd.DataFrame({"timestamp": [100, 200]})
        val_df = pd.DataFrame({"timestamp": [300, 400]})
        test_df = pd.DataFrame({"other_col": [500, 600]})

        with pytest.raises(ValueError, match="Test split missing 'timestamp'"):
            verify_temporal_boundaries(train_df, val_df, test_df, verbose=False)

    def test_train_val_overlap_raises_error(self):
        """Train/val overlap (train_max > val_min) should raise ValueError."""
        train_df = pd.DataFrame({"timestamp": [100, 200, 500]})  # max=500
        val_df = pd.DataFrame({"timestamp": [300, 400]})  # min=300
        test_df = pd.DataFrame({"timestamp": [600, 700]})

        with pytest.raises(ValueError, match="Train/val overlap"):
            verify_temporal_boundaries(train_df, val_df, test_df, verbose=False)

    def test_train_val_exact_boundary_raises_error(self):
        """Train/val exact boundary (train_max == val_min) is temporal leakage."""
        train_df = pd.DataFrame({"timestamp": [100, 200, 300]})  # max=300
        val_df = pd.DataFrame({"timestamp": [300, 400]})  # min=300 (same!)
        test_df = pd.DataFrame({"timestamp": [500, 600]})

        with pytest.raises(ValueError, match="Train/val overlap"):
            verify_temporal_boundaries(train_df, val_df, test_df, verbose=False)

    def test_val_test_overlap_raises_error(self):
        """Val/test overlap (val_max > test_min) should raise ValueError."""
        train_df = pd.DataFrame({"timestamp": [100, 200]})
        val_df = pd.DataFrame({"timestamp": [300, 600]})  # max=600
        test_df = pd.DataFrame({"timestamp": [500, 700]})  # min=500

        with pytest.raises(ValueError, match="Val/test overlap"):
            verify_temporal_boundaries(train_df, val_df, test_df, verbose=False)

    def test_val_test_exact_boundary_raises_error(self):
        """Val/test exact boundary (val_max == test_min) is temporal leakage."""
        train_df = pd.DataFrame({"timestamp": [100, 200]})
        val_df = pd.DataFrame({"timestamp": [300, 400]})  # max=400
        test_df = pd.DataFrame({"timestamp": [400, 500]})  # min=400 (same!)

        with pytest.raises(ValueError, match="Val/test overlap"):
            verify_temporal_boundaries(train_df, val_df, test_df, verbose=False)

    def test_single_row_splits_valid(self):
        """Single-row splits with valid boundaries should pass."""
        train_df = pd.DataFrame({"timestamp": [100]})
        val_df = pd.DataFrame({"timestamp": [200]})
        test_df = pd.DataFrame({"timestamp": [300]})

        result = verify_temporal_boundaries(train_df, val_df, test_df, verbose=False)

        assert result["train"] == (100, 100)
        assert result["val"] == (200, 200)
        assert result["test"] == (300, 300)

    def test_verbose_logging(self, caplog):
        """Verbose mode should log boundary information."""
        train_df = pd.DataFrame({"timestamp": [100, 200]})
        val_df = pd.DataFrame({"timestamp": [300, 400]})
        test_df = pd.DataFrame({"timestamp": [500, 600]})

        verify_temporal_boundaries(train_df, val_df, test_df, verbose=True)

        assert "Temporal boundaries verified" in caplog.text

    def test_returns_int_timestamps(self):
        """Boundary values should be integers, not numpy types."""
        train_df = pd.DataFrame({"timestamp": [100, 200]})
        val_df = pd.DataFrame({"timestamp": [300, 400]})
        test_df = pd.DataFrame({"timestamp": [500, 600]})

        result = verify_temporal_boundaries(train_df, val_df, test_df, verbose=False)

        for split_name, (start, end) in result.items():
            assert isinstance(start, int), f"{split_name} start is not int"
            assert isinstance(end, int), f"{split_name} end is not int"

    def test_millisecond_timestamps(self):
        """Should handle millisecond timestamps (real-world format)."""
        # Real timestamps: 2023-01-01, 2023-06-01, 2023-12-01
        train_df = pd.DataFrame({"timestamp": [1672531200000, 1672617600000]})
        val_df = pd.DataFrame({"timestamp": [1685577600000, 1685664000000]})
        test_df = pd.DataFrame({"timestamp": [1701388800000, 1701475200000]})

        result = verify_temporal_boundaries(train_df, val_df, test_df, verbose=False)

        assert result["train"][0] == 1672531200000
        assert result["test"][1] == 1701475200000

    def test_empty_check_before_column_check(self):
        """Empty split error should appear before missing column error."""
        # Empty df without timestamp column
        train_df = pd.DataFrame({"other": []})
        val_df = pd.DataFrame({"timestamp": [100]})
        test_df = pd.DataFrame({"timestamp": [200]})

        # Should raise "empty" not "missing column"
        with pytest.raises(ValueError, match="Train split is empty"):
            verify_temporal_boundaries(train_df, val_df, test_df, verbose=False)


class TestCreateTemporalSplits:
    """Tests for create_temporal_splits function."""

    def test_default_ratios_70_10_20(self):
        """Default ratios should produce 70/10/20 split."""
        df = pd.DataFrame(
            {
                "timestamp": list(range(100)),
                "data": list(range(100)),
            }
        )

        train, val, test = create_temporal_splits(df, save=False, verbose=False)

        assert len(train) == 70
        assert len(val) == 10
        assert len(test) == 20

    def test_preserves_temporal_order(self):
        """Train timestamps should precede val, val should precede test."""
        df = pd.DataFrame(
            {
                "timestamp": [500, 100, 300, 200, 400, 600, 700, 800, 900, 1000],
                "data": list(range(10)),
            }
        )

        train, val, test = create_temporal_splits(df, save=False, verbose=False)

        assert train["timestamp"].max() < val["timestamp"].min()
        assert val["timestamp"].max() < test["timestamp"].min()

    def test_custom_ratios(self):
        """Custom ratios should be respected."""
        df = pd.DataFrame(
            {
                "timestamp": list(range(100)),
                "data": list(range(100)),
            }
        )

        train, val, test = create_temporal_splits(
            df, train_ratio=0.5, val_ratio=0.3, save=False, verbose=False
        )

        assert len(train) == 50
        assert len(val) == 30
        assert len(test) == 20

    def test_floating_point_bug_fixed(self):
        """0.7 + 0.1 floating point issue should not lose samples."""
        # At n=10, the floating point bug would give val_end=7 instead of 8
        df = pd.DataFrame(
            {
                "timestamp": list(range(10)),
                "data": list(range(10)),
            }
        )

        train, val, test = create_temporal_splits(df, save=False, verbose=False)

        # With round(), we get correct sizes
        assert len(train) == 7
        assert len(val) == 1
        assert len(test) == 2
        # Total should equal original
        assert len(train) + len(val) + len(test) == 10

    def test_empty_dataframe_raises_error(self):
        """Empty DataFrame should raise ValueError."""
        df = pd.DataFrame({"timestamp": []})

        with pytest.raises(ValueError, match="DataFrame is empty"):
            create_temporal_splits(df, save=False, verbose=False)

    def test_missing_timestamp_column_raises_error(self):
        """Missing timestamp column should raise ValueError."""
        df = pd.DataFrame({"other_col": [1, 2, 3]})

        with pytest.raises(ValueError, match="missing 'timestamp' column"):
            create_temporal_splits(df, save=False, verbose=False)

    def test_negative_train_ratio_raises_error(self):
        """Negative train_ratio should raise ValueError."""
        df = pd.DataFrame({"timestamp": list(range(10))})

        with pytest.raises(ValueError, match="train_ratio must be between 0 and 1"):
            create_temporal_splits(df, train_ratio=-0.2, save=False, verbose=False)

    def test_train_ratio_greater_than_1_raises_error(self):
        """train_ratio > 1 should raise ValueError."""
        df = pd.DataFrame({"timestamp": list(range(10))})

        with pytest.raises(ValueError, match="train_ratio must be between 0 and 1"):
            create_temporal_splits(df, train_ratio=1.5, save=False, verbose=False)

    def test_negative_val_ratio_raises_error(self):
        """Negative val_ratio should raise ValueError."""
        df = pd.DataFrame({"timestamp": list(range(10))})

        with pytest.raises(ValueError, match="val_ratio must be between 0 and 1"):
            create_temporal_splits(df, val_ratio=-0.1, save=False, verbose=False)

    def test_val_ratio_greater_than_1_raises_error(self):
        """val_ratio > 1 should raise ValueError."""
        df = pd.DataFrame({"timestamp": list(range(10))})

        with pytest.raises(ValueError, match="val_ratio must be between 0 and 1"):
            create_temporal_splits(df, val_ratio=1.5, save=False, verbose=False)

    def test_ratios_sum_greater_than_1_raises_error(self):
        """train_ratio + val_ratio > 1 should raise ValueError."""
        df = pd.DataFrame({"timestamp": list(range(10))})

        with pytest.raises(ValueError, match="train_ratio \\+ val_ratio must be <= 1"):
            create_temporal_splits(
                df, train_ratio=0.8, val_ratio=0.5, save=False, verbose=False
            )

    def test_ratios_sum_exactly_1_valid(self):
        """train_ratio + val_ratio = 1 is valid (empty test set)."""
        df = pd.DataFrame({"timestamp": list(range(10))})

        train, val, test = create_temporal_splits(
            df, train_ratio=0.7, val_ratio=0.3, save=False, verbose=False
        )

        assert len(train) == 7
        assert len(val) == 3
        assert len(test) == 0

    def test_zero_train_ratio_valid(self):
        """train_ratio=0 is valid (all data in val/test)."""
        df = pd.DataFrame({"timestamp": list(range(10))})

        train, val, test = create_temporal_splits(
            df, train_ratio=0.0, val_ratio=0.5, save=False, verbose=False
        )

        assert len(train) == 0
        assert len(val) == 5
        assert len(test) == 5

    def test_zero_val_ratio_valid(self):
        """val_ratio=0 is valid (no validation set)."""
        df = pd.DataFrame({"timestamp": list(range(10))})

        train, val, test = create_temporal_splits(
            df, train_ratio=0.7, val_ratio=0.0, save=False, verbose=False
        )

        assert len(train) == 7
        assert len(val) == 0
        assert len(test) == 3

    def test_warns_on_empty_val_split(self, caplog):
        """Should warn when validation split ends up empty."""
        # With n=3 and default ratios, val might be empty
        df = pd.DataFrame({"timestamp": [1, 2, 3]})

        create_temporal_splits(
            df, train_ratio=0.9, val_ratio=0.05, save=False, verbose=False
        )

        assert "Validation split is empty" in caplog.text

    def test_warns_on_empty_test_split(self, caplog):
        """Should warn when test split ends up empty."""
        df = pd.DataFrame({"timestamp": list(range(10))})

        create_temporal_splits(
            df, train_ratio=0.7, val_ratio=0.3, save=False, verbose=False
        )

        assert "Test split is empty" in caplog.text

    def test_saves_to_disk_when_requested(self, tmp_path, monkeypatch):
        """Should save splits to parquet files when save=True."""
        monkeypatch.setattr("sage.data.loader.SPLITS_DIR", tmp_path)

        df = pd.DataFrame(
            {
                "timestamp": list(range(10)),
                "data": list(range(10)),
            }
        )

        create_temporal_splits(df, save=True, verbose=False)

        assert (tmp_path / "train.parquet").exists()
        assert (tmp_path / "val.parquet").exists()
        assert (tmp_path / "test.parquet").exists()

    def test_verbose_logs_split_sizes(self, caplog):
        """Verbose mode should log split sizes."""
        df = pd.DataFrame({"timestamp": list(range(100))})

        create_temporal_splits(df, save=False, verbose=True)

        assert "Train:" in caplog.text
        assert "Val:" in caplog.text
        assert "Test:" in caplog.text

    def test_all_data_preserved(self):
        """Total rows across splits should equal original."""
        df = pd.DataFrame(
            {
                "timestamp": list(range(1000)),
                "data": list(range(1000)),
            }
        )

        train, val, test = create_temporal_splits(df, save=False, verbose=False)

        assert len(train) + len(val) + len(test) == len(df)

    def test_no_data_leakage_across_splits(self):
        """No row should appear in multiple splits."""
        df = pd.DataFrame(
            {
                "timestamp": list(range(100)),
                "id": [f"row_{i}" for i in range(100)],
            }
        )

        train, val, test = create_temporal_splits(df, save=False, verbose=False)

        train_ids = set(train["id"])
        val_ids = set(val["id"])
        test_ids = set(test["id"])

        assert train_ids.isdisjoint(val_ids)
        assert train_ids.isdisjoint(test_ids)
        assert val_ids.isdisjoint(test_ids)


class TestFilter5Core:
    """Tests for filter_5_core function."""

    def test_basic_filtering(self):
        """Users and items with < min_interactions are removed."""
        # Create data where some users/items have < 5 interactions
        df = pd.DataFrame(
            {
                "user_id": ["u1"] * 10 + ["u2"] * 3 + ["u3"] * 7,  # u2 has only 3
                "parent_asin": ["p1"] * 8 + ["p2"] * 2 + ["p3"] * 10,  # p2 has only 2
            }
        )

        result = filter_5_core(df, min_interactions=5)

        # u2 and p2 should be removed
        assert "u2" not in result["user_id"].values
        assert "p2" not in result["parent_asin"].values

    def test_convergence_required(self):
        """Filtering iterates until no more removals possible."""
        # User u1 has 5 interactions, but 3 are with p2 which gets removed
        # After p2 removal, u1 only has 2 left → also removed
        df = pd.DataFrame(
            {
                "user_id": ["u1"] * 5 + ["u2"] * 10,
                "parent_asin": ["p1"] * 2 + ["p2"] * 3 + ["p1"] * 10,  # p2 has only 3
            }
        )

        result = filter_5_core(df, min_interactions=5)

        # p2 removed (only 3) → u1 now has only 2 with p1 → u1 removed
        assert "u1" not in result["user_id"].values
        assert "p2" not in result["parent_asin"].values

    def test_empty_input_returns_empty(self):
        """Empty DataFrame returns empty DataFrame."""
        df = pd.DataFrame({"user_id": [], "parent_asin": []})

        result = filter_5_core(df)

        assert result.empty

    def test_all_filtered_out_returns_empty(self):
        """When all users/items have < min_interactions, returns empty."""
        df = pd.DataFrame(
            {
                "user_id": ["u1", "u2", "u3"],
                "parent_asin": ["p1", "p2", "p3"],
            }
        )

        result = filter_5_core(df, min_interactions=5)

        assert result.empty

    def test_min_interactions_1_keeps_all(self):
        """min_interactions=1 keeps all data."""
        df = pd.DataFrame(
            {
                "user_id": ["u1", "u2", "u3"],
                "parent_asin": ["p1", "p2", "p3"],
            }
        )

        result = filter_5_core(df, min_interactions=1)

        assert len(result) == len(df)

    def test_preserves_other_columns(self):
        """Non-filter columns are preserved."""
        df = pd.DataFrame(
            {
                "user_id": ["u1"] * 10,
                "parent_asin": ["p1"] * 10,
                "rating": [4.0] * 10,
                "text": ["review"] * 10,
            }
        )

        result = filter_5_core(df, min_interactions=5)

        assert "rating" in result.columns
        assert "text" in result.columns
        assert result["rating"].iloc[0] == 4.0

    def test_resets_index(self):
        """Result has clean 0-based index."""
        df = pd.DataFrame(
            {
                "user_id": ["u1"] * 10,
                "parent_asin": ["p1"] * 10,
            }
        )
        df.index = range(100, 110)  # Non-zero starting index

        result = filter_5_core(df, min_interactions=5)

        assert list(result.index) == list(range(len(result)))

    def test_logs_retention_stats(self, caplog):
        """Should log retention percentage."""
        df = pd.DataFrame(
            {
                "user_id": ["u1"] * 10 + ["u2"] * 10,
                "parent_asin": ["p1"] * 10 + ["p2"] * 10,
            }
        )

        filter_5_core(df, min_interactions=5)

        assert "retained" in caplog.text
        assert "%" in caplog.text

    def test_logs_warning_when_all_filtered(self, caplog):
        """Should warn when all data is filtered out."""
        df = pd.DataFrame(
            {
                "user_id": ["u1", "u2", "u3"],
                "parent_asin": ["p1", "p2", "p3"],
            }
        )

        filter_5_core(df, min_interactions=5)

        assert "All data filtered out" in caplog.text

    def test_exact_threshold_kept(self):
        """Users/items with exactly min_interactions are kept."""
        df = pd.DataFrame(
            {
                "user_id": ["u1"] * 5 + ["u2"] * 5,  # Both have exactly 5
                "parent_asin": ["p1"] * 5 + ["p2"] * 5,  # Both have exactly 5
            }
        )

        result = filter_5_core(df, min_interactions=5)

        assert len(result) == 10
        assert set(result["user_id"]) == {"u1", "u2"}
        assert set(result["parent_asin"]) == {"p1", "p2"}

    def test_large_min_interactions(self):
        """Large min_interactions filters aggressively."""
        df = pd.DataFrame(
            {
                "user_id": ["u1"] * 50 + ["u2"] * 10,
                "parent_asin": ["p1"] * 30 + ["p2"] * 30,
            }
        )

        result = filter_5_core(df, min_interactions=20)

        # Only u1 (50) and p1 (30 from u1) survive
        # But wait - p1 only has 30, and after u2 is filtered,
        # p1's count from u1 depends on which product u1 interacted with
        # Let me recalculate: u1 has 50 interactions (all on p1 if we look at pattern)
        # Actually the pattern shows first 50 are u1, split across p1(30) and p2(30)
        # So u1 has 50 total, u2 has 10 total
        # p1 has 30, p2 has 30
        # With min=20: u2(10) removed. After u2 removal, p1 still has ~30, p2 still has ~30
        # Actually the indexing: u1*50 maps to p1*30 + p2*30 means u1 gets 30 on p1, 20 on p2
        # and u2 gets 0 on p1, 10 on p2
        # So p1=30 (all u1), p2=30 (20 u1 + 10 u2)
        # Remove u2 (10 interactions) → p2 now has 20
        # All survive with min=20
        assert len(result) > 0

    def test_handles_single_user(self):
        """Single user with enough items works."""
        df = pd.DataFrame(
            {
                "user_id": ["u1"] * 10,
                "parent_asin": [f"p{i}" for i in range(10)],  # 10 different items
            }
        )

        result = filter_5_core(df, min_interactions=5)

        # u1 has 10 interactions, but each item only has 1
        # Items get filtered → u1 gets filtered
        assert result.empty

    def test_dense_interaction_matrix(self):
        """Dense data where everyone interacts with everything."""
        users = ["u1", "u2", "u3"]
        items = ["p1", "p2", "p3"]
        # Each user interacts with each item twice = 6 per user, 6 per item
        data = []
        for u in users:
            for p in items:
                data.extend([{"user_id": u, "parent_asin": p}] * 2)

        df = pd.DataFrame(data)

        result = filter_5_core(df, min_interactions=5)

        # 6 interactions per user, 6 per item - all kept
        assert len(result) == len(df)
