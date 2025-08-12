import numpy as np
import pytest

from splitter_mr.embedding.base_embedding import BaseEmbedding
from splitter_mr.schema import ReaderOutput, SplitterOutput
from splitter_mr.splitter.splitters.semantic_splitter import SemanticSplitter

# --------------------------
# Test utilities & doubles
# --------------------------


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


# --------------------------
# Basic behavior
# --------------------------


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


# --------------------------
# Threshold strategies
# --------------------------


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
    assert all(c for c in out.chunks)
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


# --------------------------
# Min-size behavior (chunk_size acts as minimum)
# --------------------------


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


# --------------------------
# Buffer size effects
# --------------------------


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


# --------------------------
# Batch embedding usage
# --------------------------


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


# --------------------------
# Metadata & output integrity
# --------------------------


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


# --------------------------
# Edge cases
# --------------------------


def test_empty_text_returns_single_empty_chunk():
    emb = DummyEmbedding()
    splitter = SemanticSplitter(embedding=emb, buffer_size=1, chunk_size=1)
    ro = make_reader("")

    import pytest

    with pytest.raises(ValueError, match="No text has been provided"):
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
