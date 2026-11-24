"""Hand-written cosine similarity of scripts/compare.py."""

from __future__ import annotations

import math

import pytest

from scripts.compare import cosine_similarity


def test_identical_vectors_score_one():
    assert cosine_similarity([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == pytest.approx(1.0)


def test_length_does_not_matter_only_direction():
    # NOTA: [2,4,6] es [1,2,3] estirado: misma dirección, coseno 1.
    assert cosine_similarity([1.0, 2.0, 3.0], [2.0, 4.0, 6.0]) == pytest.approx(1.0)


def test_orthogonal_and_opposite():
    assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)
    assert cosine_similarity([1.0, 0.0], [-1.0, 0.0]) == pytest.approx(-1.0)


def test_45_degrees():
    assert cosine_similarity([1.0, 0.0], [1.0, 1.0]) == pytest.approx(math.sqrt(2) / 2)


def test_zero_vector_returns_zero():
    assert cosine_similarity([0.0, 0.0], [1.0, 1.0]) == 0.0


def test_different_lengths_raise():
    with pytest.raises(ValueError):
        cosine_similarity([1.0], [1.0, 2.0])
