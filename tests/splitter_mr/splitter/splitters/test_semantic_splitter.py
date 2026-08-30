from typing import Any, Dict, List

import numpy as np
import pytest

from splitter_mr.embedding.base_embedding import BaseEmbedding
from splitter_mr.schema import (
    ReaderOutput,
    ReaderOutputException,
    SplitterConfigException,
    SplitterInputWarning,
    SplitterOutput,
    SplitterOutputException,
    SplitterOutputWarning,
)
from splitter_mr.splitter.splitters.semantic_splitter import SemanticSplitter

# ---- Mocks, fixtures and helpers ---- #


class DummyEmbedding(BaseEmbedding):
    """
    Deterministic, fast embedding for tests.
    - Tokenizes on whitespace.
    - Hashes each token to a fixed dim (16).
    - Sums one-hot vectors, then L2-normalizes.
    """

    def __init__(self, model_name: str = "dummy-emb-16", dim: int = 16) -> None:
        self.model_name = model_name
        self.dim = dim
        self._embed_text_calls = 0
        self._embed_docs_calls = 0

    def get_client(self):
        return None

    def _vec_for_tokens(self, toks):
        v = np.zeros(self.dim, dtype=np.float64)
        for t in toks:
            if not t:
                continue
            idx = (hash(t) % self.dim + self.dim) % self.dim
            v[idx] += 1.0
        n = np.linalg.norm(v)
        return (v / n).tolist() if n > 0 else v.tolist()

    def embed_text(self, text: str, **parameters):
        self._embed_text_calls += 1
        toks = str(text).lower().split()
        return self._vec_for_tokens(toks)

    def embed_documents(self, texts, **parameters):
        self._embed_docs_calls += 1
        out = []
        for t in texts:
            toks = str(t).lower().split()
            out.append(self._vec_for_tokens(toks))
        return out


def make_reader(
    text: str, name: str = "doc.txt", path: str = "/tmp/doc.txt"
) -> ReaderOutput:
    return ReaderOutput(
        text=text,
        document_name=name,
        document_path=path,
        conversion_method="txt",
        reader_method="vanilla",
        ocr_method=None,
        page_placeholder=None,
        metadata={"source": "unit-test"},
    )


class FailingEmbedding(BaseEmbedding):
    def __init__(self, model_name: str = "failing-emb") -> None:
        # minimal concrete init for tests
        self.model_name = model_name

    def get_client(self) -> Any:
        return None

    def embed_text(self, text: str, **parameters: Dict[str, Any]) -> List[float]:
        # Should never be called in this test
        raise RuntimeError("embed_text should not be used")

    def embed_documents(
        self, texts: List[str], **parameters: Dict[str, Any]
    ) -> List[List[float]]:
        # Force failure to exercise error handling
        raise RuntimeError("embedding boom")


class WrongShapeEmbedding(BaseEmbedding):
    def __init__(self, model_name: str = "wrong-shape-emb") -> None:
        self.model_name = model_name

    def get_client(self) -> Any:
        return None

    def embed_text(self, text: str, **parameters: Dict[str, Any]) -> List[float]:
        return [0.0]

    def embed_documents(
        self, texts: List[str], **parameters: Dict[str, Any]
    ) -> List[List[float]]:
        # Return fewer embeddings than windows to trigger shape mismatch
        return [[0.0] for _ in range(max(0, len(texts) - 1))]


class NaNEmbedding(BaseEmbedding):
    def __init__(self, model_name: str = "nan-emb") -> None:
        self.model_name = model_name

    def get_client(self) -> Any:
        return None

    def embed_text(self, text: str, **parameters: Dict[str, Any]) -> List[float]:
        return [float("nan")]

    def embed_documents(
        self, texts: List[str], **parameters: Dict[str, Any]
    ) -> List[List[float]]:
        # Each embedding contains NaN; should trigger non-finite distance check
        return [[float("nan")] for _ in texts]


# ---- Test cases ---- #


def test_single_sentence_returns_whole_text():
    emb = DummyEmbedding()
    splitter = SemanticSplitter(embedding=emb, buffer_size=1, chunk_size=1)
    ro = make_reader("Hello world.")
    out = splitter.split(ro)

    assert isinstance(out, SplitterOutput)
    assert out.chunks == ["Hello world."]
    assert len(out.chunk_id) == 1
    assert out.document_name == "doc.txt"
    assert out.metadata is not None


def test_two_sentences_gradient_mode_returns_both():
    emb = DummyEmbedding()
    splitter = SemanticSplitter(
        embedding=emb,
        buffer_size=1,
        breakpoint_threshold_type="gradient",
        chunk_size=1,
    )
    ro = make_reader("Cats purr. Dogs bark.")
    out = splitter.split(ro)

    # With gradient mode & exactly 2 sentences, we should bypass gradient calc
    assert out.chunks == ["Cats purr.", "Dogs bark."]


# Threshold strategies


@pytest.mark.parametrize(
    "strategy", ["percentile", "standard_deviation", "interquartile"]
)
def test_threshold_strategies_do_not_crash_and_produce_chunks(strategy):
    text = (
        "Cats purr. Cats like naps. "
        "Dogs bark. Dogs fetch. "
        "Stocks rally. Markets fall. Bonds rise."
    )
    emb = DummyEmbedding()
    splitter = SemanticSplitter(
        embedding=emb,
        buffer_size=1,
        breakpoint_threshold_type=strategy,
        # lower min size to ensure we don't skip everything
        chunk_size=5,
    )
    ro = make_reader(text)
    out = splitter.split(ro)

    assert len(out.chunks) >= 1
    # Ensure chunks do not exceed original text length and are non-empty
    assert all(out.chunks)
    assert "".join(out.chunks).replace(" ", "") in text.replace(" ", "")


def test_percentile_amount_controls_splits():
    # Force a less extreme threshold so a split is likely
    text = "Cats purr. Cats sleep. Dogs bark. Dogs fetch. Markets rally."
    emb = DummyEmbedding()
    splitter = SemanticSplitter(
        embedding=emb,
        buffer_size=1,
        breakpoint_threshold_type="percentile",
        breakpoint_threshold_amount=50.0,  # median threshold
        chunk_size=1,
    )
    ro = make_reader(text)
    out = splitter.split(ro)

    assert len(out.chunks) >= 2


def test_number_of_chunks_targets_approximate_count():
    # Provide enough variety to create several distance spikes
    text = (
        "Cats purr. Cats sleep. "
        "Dogs bark. Dogs fetch. "
        "Birds chirp. Birds fly. "
        "Stocks rally. Markets fall. "
        "Bonds rise. Commodities surge."
    )
    emb = DummyEmbedding()
    splitter = SemanticSplitter(
        embedding=emb,
        buffer_size=1,
        number_of_chunks=3,  # aim for 3 chunks
        chunk_size=1,
    )
    ro = make_reader(text)
    out = splitter.split(ro)

    # It's an approximation; ensure we get a sensible number near target.
    assert 2 <= len(out.chunks) <= 4


# Min-size behavior (chunk_size acts as minimum)


def test_min_size_merges_small_chunks():
    # With a big min size (chunk_size), small sentence groups get merged
    text = "A. B. C. D. E. F."
    emb = DummyEmbedding()
    splitter = SemanticSplitter(
        embedding=emb,
        buffer_size=0,
        breakpoint_threshold_type="percentile",
        breakpoint_threshold_amount=50.0,
        chunk_size=len(text) + 10,  # larger than total text => one chunk
    )
    ro = make_reader(text)
    out = splitter.split(ro)

    assert len(out.chunks) == 1
    assert out.chunks[0].replace(" ", "") == text.replace(" ", "")


# Buffer size effects


def test_buffer_size_changes_windows():
    text = (
        "Alpha beta. Alpha gamma. Delta epsilon. Delta zeta. Theta iota. Kappa lambda."
    )
    emb = DummyEmbedding()
    ro = make_reader(text)

    # No buffer
    s0 = SemanticSplitter(embedding=emb, buffer_size=0, chunk_size=1)
    out0 = s0.split(ro)

    # With buffer
    s1 = SemanticSplitter(embedding=emb, buffer_size=1, chunk_size=1)
    out1 = s1.split(ro)

    # Both runs should succeed and return at least one chunk
    assert len(out0.chunks) >= 1
    assert len(out1.chunks) >= 1


# Batch embedding usage


def test_batch_embedding_is_used(monkeypatch):
    emb = DummyEmbedding()

    # Track calls explicitly
    def spy_embed_documents(texts, **kwargs):
        emb._embed_docs_calls += 1
        return DummyEmbedding().embed_documents(texts)

    # Patch only embed_documents; embed_text should NOT be used by splitter
    monkeypatch.setattr(emb, "embed_documents", spy_embed_documents, raising=True)

    splitter = SemanticSplitter(embedding=emb, buffer_size=1, chunk_size=1)
    ro = make_reader("One. Two. Three. Four.")
    out = splitter.split(ro)

    assert len(out.chunks) >= 1
    assert emb._embed_docs_calls == 1


def test_embed_text_not_called_when_batch_available(monkeypatch):
    emb = DummyEmbedding()

    # Make embed_text raise if called (we expect only embed_documents)
    def bomb(*args, **kwargs):
        raise AssertionError("embed_text should not be called by SemanticSplitter")

    monkeypatch.setattr(emb, "embed_text", bomb, raising=True)

    splitter = SemanticSplitter(embedding=emb, buffer_size=1, chunk_size=1)
    ro = make_reader("First sentence. Second sentence. Third sentence.")
    out = splitter.split(ro)

    assert len(out.chunks) >= 1


# Metadata & output integrity


def test_output_metadata_is_propagated():
    emb = DummyEmbedding()
    splitter = SemanticSplitter(embedding=emb, buffer_size=1, chunk_size=1)
    ro = make_reader("Alpha. Beta. Gamma.", name="sample.txt", path="/tmp/sample.txt")
    out = splitter.split(ro)

    assert out.document_name == "sample.txt"
    assert out.document_path == "/tmp/sample.txt"
    assert out.reader_method == "vanilla"
    assert out.conversion_method == "txt"
    assert out.split_method == "semantic_splitter"
    assert "buffer_size" in out.split_params
    assert "model_name" in out.split_params
    assert out.metadata is not None


# Edge cases


def test_empty_text_raises_reader_output_exception():
    emb = DummyEmbedding()
    splitter = SemanticSplitter(embedding=emb, buffer_size=1, chunk_size=1)
    ro = make_reader("")

    with pytest.raises(ReaderOutputException, match="empty or None"):
        splitter.split(ro)


def test_gradient_strategy_path_executes():
    text = "One. Two. Three. Four. Five."
    emb = DummyEmbedding()
    splitter = SemanticSplitter(
        embedding=emb,
        buffer_size=1,
        breakpoint_threshold_type="gradient",
        chunk_size=1,
    )
    ro = make_reader(text)
    out = splitter.split(ro)

    assert len(out.chunks) >= 1


def test_interquartile_strategy_path_executes():
    text = "Cat. Cat. Dog. Dog. Market. Market."
    emb = DummyEmbedding()
    splitter = SemanticSplitter(
        embedding=emb,
        buffer_size=1,
        breakpoint_threshold_type="interquartile",
        chunk_size=1,
    )
    ro = make_reader(text)
    out = splitter.split(ro)

    assert len(out.chunks) >= 1


# --- Error & warning handling tests --- #

# Config-time validation


def test_invalid_embedding_backend_raises_config_exception():
    class BadEmbedding:
        # No embed_documents method
        pass

    with pytest.raises(SplitterConfigException, match="embed_documents"):
        SemanticSplitter(embedding=BadEmbedding(), buffer_size=1, chunk_size=1)


def test_negative_buffer_size_raises_config_exception():
    emb = DummyEmbedding()
    with pytest.raises(SplitterConfigException, match="buffer_size must be >= 0"):
        SemanticSplitter(embedding=emb, buffer_size=-1, chunk_size=1)


def test_invalid_breakpoint_type_raises_config_exception():
    emb = DummyEmbedding()
    with pytest.raises(
        SplitterConfigException, match="Invalid breakpoint_threshold_type"
    ):
        SemanticSplitter(
            embedding=emb,
            buffer_size=1,
            breakpoint_threshold_type="not-a-strategy",
            chunk_size=1,
        )


def test_invalid_percentile_amount_raises_config_exception():
    emb = DummyEmbedding()
    # 123.0 is outside [0, 100] and not in (0,1]
    with pytest.raises(SplitterConfigException, match="breakpoint_threshold_amount"):
        SemanticSplitter(
            embedding=emb,
            buffer_size=1,
            breakpoint_threshold_type="percentile",
            breakpoint_threshold_amount=123.0,
            chunk_size=1,
        )


def test_number_of_chunks_must_be_positive_and_finite():
    emb = DummyEmbedding()
    with pytest.raises(SplitterConfigException, match="positive finite"):
        SemanticSplitter(embedding=emb, buffer_size=1, number_of_chunks=0, chunk_size=1)

    with pytest.raises(SplitterConfigException, match="positive finite"):
        SemanticSplitter(
            embedding=emb,
            buffer_size=1,
            number_of_chunks=float("inf"),
            chunk_size=1,
        )


def test_ratio_percentile_amount_emits_input_warning():
    emb = DummyEmbedding()
    # 0.8 in (0,1] should be treated as ratio and scaled, with a warning
    with pytest.warns(SplitterInputWarning, match="ratio"):
        splitter = SemanticSplitter(
            embedding=emb,
            buffer_size=1,
            breakpoint_threshold_type="percentile",
            breakpoint_threshold_amount=0.8,
            chunk_size=1,
        )
    # sanity: still works
    ro = make_reader("A. B. C.")
    out = splitter.split(ro)
    assert len(out.chunks) >= 1


def test_non_integer_number_of_chunks_emits_input_warning():
    emb = DummyEmbedding()
    with pytest.warns(SplitterInputWarning, match="not an integer"):
        splitter = SemanticSplitter(
            embedding=emb,
            buffer_size=1,
            number_of_chunks=3.7,
            chunk_size=1,
        )
    ro = make_reader("A. B. C. D.")
    out = splitter.split(ro)
    assert len(out.chunks) >= 1


# Runtime embedding errors & distances


def test_embedding_backend_failure_raises_splitter_output_exception():
    emb = FailingEmbedding()
    splitter = SemanticSplitter(embedding=emb, buffer_size=1, chunk_size=1)
    ro = make_reader("One. Two. Three.")
    with pytest.raises(SplitterOutputException, match="Embedding backend failed"):
        splitter.split(ro)


def test_embedding_shape_mismatch_raises_splitter_output_exception():
    emb = WrongShapeEmbedding()
    splitter = SemanticSplitter(embedding=emb, buffer_size=1, chunk_size=1)
    ro = make_reader("One. Two. Three.")
    with pytest.raises(
        SplitterOutputException, match="does not match the number of windows"
    ):
        splitter.split(ro)


def test_non_finite_distances_raise_splitter_output_exception():
    emb = NaNEmbedding()
    splitter = SemanticSplitter(embedding=emb, buffer_size=1, chunk_size=1)
    ro = make_reader("One. Two. Three.")
    with pytest.raises(
        SplitterOutputException, match="Non-finite values .* semantic distances"
    ):
        splitter.split(ro)


def test_sentence_splitter_failure_is_wrapped_in_splitter_output_exception(monkeypatch):
    emb = DummyEmbedding()
    splitter = SemanticSplitter(embedding=emb, buffer_size=1, chunk_size=1)
    ro = make_reader("Hello. World.")

    def boom(*_args, **_kwargs):
        raise RuntimeError("sentence splitting boom")

    # Patch the underlying SentenceSplitter, so the wrapper in
    # SemanticSplitter._split_into_sentences catches and re-raises.
    monkeypatch.setattr(splitter._sentence_splitter, "split", boom, raising=True)

    with pytest.raises(SplitterOutputException, match="Sentence splitting failed"):
        splitter.split(ro)


# Runtime warnings (SplitterOutputWarning)


def test_number_of_chunks_larger_than_possible_emits_output_warning():
    text = "A. B."
    emb = DummyEmbedding()
    splitter = SemanticSplitter(
        embedding=emb,
        buffer_size=1,
        number_of_chunks=10,  # larger than max possible (=2)
        chunk_size=1,
    )
    ro = make_reader(text)

    with pytest.warns(SplitterOutputWarning, match="larger than the maximum possible"):
        out = splitter.split(ro)

    # Still returns sensible chunks
    assert 1 <= len(out.chunks) <= 2


def test_no_breakpoints_emits_output_warning(monkeypatch):
    text = "A. B. C."
    emb = DummyEmbedding()
    splitter = SemanticSplitter(
        embedding=emb,
        buffer_size=1,
        breakpoint_threshold_type="percentile",
        breakpoint_threshold_amount=50.0,
        chunk_size=1,
    )
    ro = make_reader(text)

    # Force _calculate_breakpoint_threshold to choose a threshold above all values
    def fake_calc_threshold(self, distances):
        # threshold greater than any value in ref array -> no indices_above
        return 1.0, [0.0 for _ in distances]

    monkeypatch.setattr(
        SemanticSplitter,
        "_calculate_breakpoint_threshold",
        fake_calc_threshold,
        raising=True,
    )

    with pytest.warns(
        SplitterOutputWarning, match="did not detect any semantic breakpoints"
    ):
        out = splitter.split(ro)

    assert len(out.chunks) == 1
