"""
ML Tests: Training Data Pipeline

Tests that training data:
- Has correct format for model training
- Contains valid score ranges
- Has no data leakage between splits
- Maintains consistent dimensions across splits

Run with: pytest tests/ml/test_data_pipeline.py -v
"""

import json
import pytest
from pathlib import Path


def find_training_datasets():
    """Find all training datasets in the project."""
    project_root = Path(__file__).parent.parent.parent
    datasets_dir = project_root / "datasets" / "training"

    if not datasets_dir.exists():
        return []

    # Find directories with train.jsonl
    datasets = []
    for subdir in datasets_dir.iterdir():
        if subdir.is_dir() and (subdir / "train.jsonl").exists():
            datasets.append(subdir)

    return datasets


def load_jsonl(filepath: Path):
    """Load a JSONL file."""
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


class TestTrainingDataFormat:
    """Tests for training data format validity."""

    @pytest.fixture
    def training_datasets(self):
        """Get all available training datasets."""
        datasets = find_training_datasets()
        if not datasets:
            pytest.skip("No training datasets found")
        return datasets

    def test_required_files_exist(self, training_datasets):
        """Each training dataset should have train/val/test splits."""
        for dataset_dir in training_datasets:
            assert (dataset_dir / "train.jsonl").exists(), f"Missing train.jsonl in {dataset_dir}"
            assert (dataset_dir / "val.jsonl").exists(), f"Missing val.jsonl in {dataset_dir}"
            assert (dataset_dir / "test.jsonl").exists(), f"Missing test.jsonl in {dataset_dir}"

    def test_required_fields_present(self, training_datasets):
        """Training examples should have required fields."""
        required_fields = ['id', 'title', 'content', 'labels', 'dimension_names']

        for dataset_dir in training_datasets:
            train_data = load_jsonl(dataset_dir / "train.jsonl")

            if not train_data:
                continue

            # Check first 10 examples
            for example in train_data[:10]:
                for field in required_fields:
                    assert field in example, f"Missing field '{field}' in {dataset_dir}"

    def test_labels_are_numeric_arrays(self, training_datasets):
        """Labels should be arrays of numbers."""
        for dataset_dir in training_datasets:
            train_data = load_jsonl(dataset_dir / "train.jsonl")

            for example in train_data[:20]:
                labels = example.get('labels', [])
                assert isinstance(labels, list), f"Labels should be list in {dataset_dir}"
                assert all(isinstance(l, (int, float)) for l in labels), \
                    f"All labels should be numeric in {dataset_dir}"

    def test_labels_in_valid_range(self, training_datasets):
        """All labels should be in [0, 10] range."""
        for dataset_dir in training_datasets:
            for split in ['train.jsonl', 'val.jsonl', 'test.jsonl']:
                data = load_jsonl(dataset_dir / split)

                for example in data:
                    labels = example.get('labels', [])
                    for i, score in enumerate(labels):
                        assert 0 <= score <= 10, \
                            f"Score {score} out of range in {dataset_dir}/{split}, example {example.get('id')}"

    def test_labels_array_length_matches_dimensions(self, training_datasets):
        """Labels array length should match dimension_names."""
        for dataset_dir in training_datasets:
            train_data = load_jsonl(dataset_dir / "train.jsonl")

            for example in train_data[:50]:
                labels = example.get('labels', [])
                dims = example.get('dimension_names', [])
                assert len(labels) == len(dims), \
                    f"Labels/dims mismatch: {len(labels)} vs {len(dims)} in {dataset_dir}"


class TestDataLeakage:
    """Tests to ensure no data leakage between splits."""

    @pytest.fixture
    def training_datasets(self):
        """Get all available training datasets."""
        datasets = find_training_datasets()
        if not datasets:
            pytest.skip("No training datasets found")
        return datasets

    def test_no_id_overlap_between_splits(self, training_datasets):
        """Same ID should not appear in multiple splits."""
        for dataset_dir in training_datasets:
            train_ids = {ex['id'] for ex in load_jsonl(dataset_dir / "train.jsonl")}
            val_ids = {ex['id'] for ex in load_jsonl(dataset_dir / "val.jsonl")}
            test_ids = {ex['id'] for ex in load_jsonl(dataset_dir / "test.jsonl")}

            train_val = train_ids & val_ids
            train_test = train_ids & test_ids
            val_test = val_ids & test_ids

            assert len(train_val) == 0, f"Train-Val overlap in {dataset_dir}: {list(train_val)[:5]}"
            assert len(train_test) == 0, f"Train-Test overlap in {dataset_dir}: {list(train_test)[:5]}"
            assert len(val_test) == 0, f"Val-Test overlap in {dataset_dir}: {list(val_test)[:5]}"

    def test_no_duplicate_ids_within_split(self, training_datasets):
        """No duplicate IDs within a single split."""
        for dataset_dir in training_datasets:
            for split in ['train.jsonl', 'val.jsonl', 'test.jsonl']:
                data = load_jsonl(dataset_dir / split)
                ids = [ex['id'] for ex in data]

                assert len(ids) == len(set(ids)), \
                    f"Duplicate IDs in {dataset_dir}/{split}"


class TestDataDistribution:
    """Tests for reasonable data distribution."""

    @pytest.fixture
    def training_datasets(self):
        """Get all available training datasets."""
        datasets = find_training_datasets()
        if not datasets:
            pytest.skip("No training datasets found")
        return datasets

    def test_split_ratios_reasonable(self, training_datasets):
        """Split ratios should be approximately 80/10/10."""
        for dataset_dir in training_datasets:
            train_count = len(load_jsonl(dataset_dir / "train.jsonl"))
            val_count = len(load_jsonl(dataset_dir / "val.jsonl"))
            test_count = len(load_jsonl(dataset_dir / "test.jsonl"))

            total = train_count + val_count + test_count
            if total == 0:
                continue

            train_pct = train_count / total * 100
            val_pct = val_count / total * 100
            test_pct = test_count / total * 100

            # Allow ±15% variance for small datasets
            assert 65 <= train_pct <= 95, f"Train split {train_pct:.1f}% outside bounds in {dataset_dir}"
            assert 2 <= val_pct <= 25, f"Val split {val_pct:.1f}% outside bounds in {dataset_dir}"
            assert 2 <= test_pct <= 25, f"Test split {test_pct:.1f}% outside bounds in {dataset_dir}"

    def test_score_variance_exists(self, training_datasets):
        """Scores should have some variance (not all identical)."""
        import statistics

        for dataset_dir in training_datasets:
            train_data = load_jsonl(dataset_dir / "train.jsonl")

            if len(train_data) < 10:
                continue

            # Collect all scores
            all_scores = []
            for example in train_data:
                all_scores.extend(example.get('labels', []))

            if len(all_scores) < 2:
                continue

            stdev = statistics.stdev(all_scores)
            assert stdev > 0.5, f"Very low score variance ({stdev:.2f}) in {dataset_dir} - may indicate issue"

    def test_content_not_empty(self, training_datasets):
        """Content should not be empty."""
        for dataset_dir in training_datasets:
            train_data = load_jsonl(dataset_dir / "train.jsonl")

            empty_count = 0
            for example in train_data:
                content = example.get('content', '')
                if not content or len(content.strip()) < 50:
                    empty_count += 1

            # Allow up to 5% empty/short content
            max_empty = max(1, len(train_data) * 0.05)
            assert empty_count <= max_empty, \
                f"Too many empty/short content examples ({empty_count}) in {dataset_dir}"


class TestDimensionConsistency:
    """Tests for dimension consistency across data."""

    @pytest.fixture
    def training_datasets(self):
        """Get all available training datasets."""
        datasets = find_training_datasets()
        if not datasets:
            pytest.skip("No training datasets found")
        return datasets

    def test_dimensions_consistent_across_splits(self, training_datasets):
        """Dimension names should be identical across train/val/test."""
        for dataset_dir in training_datasets:
            train_data = load_jsonl(dataset_dir / "train.jsonl")
            val_data = load_jsonl(dataset_dir / "val.jsonl")
            test_data = load_jsonl(dataset_dir / "test.jsonl")

            if not (train_data and val_data and test_data):
                continue

            train_dims = train_data[0].get('dimension_names', [])
            val_dims = val_data[0].get('dimension_names', [])
            test_dims = test_data[0].get('dimension_names', [])

            assert train_dims == val_dims, f"Train/Val dimension mismatch in {dataset_dir}"
            assert train_dims == test_dims, f"Train/Test dimension mismatch in {dataset_dir}"

    def test_all_examples_have_same_dimensions(self, training_datasets):
        """All examples in a dataset should have the same dimension names."""
        for dataset_dir in training_datasets:
            train_data = load_jsonl(dataset_dir / "train.jsonl")

            if not train_data:
                continue

            expected_dims = train_data[0].get('dimension_names', [])

            for i, example in enumerate(train_data):
                actual_dims = example.get('dimension_names', [])
                assert actual_dims == expected_dims, \
                    f"Dimension mismatch at index {i} in {dataset_dir}"
