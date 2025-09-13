import re
import uuid
from typing import Dict, Iterable, List, Pattern, Tuple, Union

from ...schema import ReaderOutput, SplitterOutput
from ..base_splitter import BaseSplitter


class KeywordSplitter(BaseSplitter):
    """
    Splitter that chunks text around *keyword* boundaries using regular expressions.

    This splitter searches the input text for one or more *keyword patterns* (regex)
    and creates chunks at each match boundary. You can control how the matched
    delimiter is attached to the resulting chunks (before/after/both/none) and apply a
    secondary, size-based re-chunking to respect ``chunk_size``.

    The splitter emits a :class:`~..schema.SplitterOutput` with metadata including
    per-keyword match counts and raw match spans.

    Args:
        patterns (Union[List[str], Dict[str, str]]): A list of regex pattern strings **or** a mapping of
            ``name -> regex pattern``. When a dict is provided, the keys are used in
            the metadata counts. When a list is provided, synthetic names are
            generated (``k0``, ``k1``, ...).
        flags (int): Standard ``re`` flags combined with ``|`` (e.g., ``re.IGNORECASE``).
        include_delimiters (str): Where to attach the matched keyword delimiter.
            One of ``"none"``, ``"before"``, ``"after"``, ``"both"``.
            - ``before`` (default) appends the match to the *preceding* chunk.
            - ``after`` prepends the match to the *following* chunk.
            - ``both`` duplicates the match on both sides.
            - ``none`` omits the delimiter from both sides.
        chunk_size (int): Target maximum size (in characters) for each chunk. When a
            produced chunk exceeds this value, it is *soft*-wrapped by whitespace
            using a greedy strategy.

    Notes:
        - All regexes are compiled into **one** alternation with *named groups* when
          ``patterns`` is a dict. This simplifies per-keyword accounting.
        - If the input text is empty or no matches are found, the entire text
          becomes a single chunk (subject to size-based re-chunking).
    """

    def __init__(
        self,
        patterns: Union[List[str], Dict[str, str]],
        *,
        flags: int = 0,
        include_delimiters: str = "before",
        chunk_size: int = 1000,
    ) -> None:
        """
        Initialize the KeywordSplitter.

        Args:
            patterns (Union[List[str], Dict[str, str]]): Keyword regex patterns.
            flags (int): Regex flags.
            include_delimiters (str): How to include delimiters (before, after, both, none).
            chunk_size (int): Max chunk size in characters.
        """
        super().__init__(chunk_size=chunk_size)
        self.include_delimiters = self._validate_include_delimiters(include_delimiters)
        self.pattern_names, self.compiled = self._compile_patterns(patterns, flags)
        self.flags = flags

    def split(self, reader_output: ReaderOutput) -> SplitterOutput:
        """
        Split ReaderOutput into keyword-delimited chunks and build structured output.

        Args:
            reader_output (ReaderOutput): Input document and metadata.

        Returns:
            SplitterOutput: Output structure with chunked text and metadata.
        """
        text = reader_output.text or ""

        # Ensure document_id is present so it propagates (fixes metadata test)
        if not reader_output.document_id:
            reader_output.document_id = str(uuid.uuid4())

        # Primary split by keyword matches (names used for counts)
        raw_chunks, match_spans, match_names = self._split_by_keywords(text)

        # Secondary size-based re-chunking to respect chunk_size
        sized_chunks: List[str] = []
        for ch in raw_chunks:
            sized_chunks.extend(self._soft_wrap(ch, self.chunk_size))
        if not sized_chunks:
            sized_chunks = [""]

        # Generate IDs
        chunk_ids = self._generate_chunk_ids(len(sized_chunks))

        # Build metadata (ensure counts/spans are always present)
        matches_meta = {
            "counts": self._count_by_name(match_names),
            "spans": match_spans,
            "include_delimiters": self.include_delimiters,
            "flags": self.flags,
            "pattern_names": self.pattern_names,
            "chunk_size": self.chunk_size,
        }

        return self._build_output(
            reader_output=reader_output,
            chunks=sized_chunks,
            chunk_ids=chunk_ids,
            matches_meta=matches_meta,
        )

    # ---- Internals ------------------------------------------------------ #

    @staticmethod
    def _validate_include_delimiters(value: str) -> str:
        """
        Validate and normalize include_delimiters argument.

        Args:
            value (str): One of {"none", "before", "after", "both"}.

        Returns:
            str: Normalized delimiter mode.

        Raises:
            ValueError: If the mode is invalid.
        """
        allowed = {"none", "before", "after", "both"}
        v = value.lower().strip()
        if v not in allowed:
            raise ValueError(
                f"include_delimiters must be one of {sorted(allowed)}, got {value!r}"
            )
        return v

    @staticmethod
    def _compile_patterns(
        patterns: Union[List[str], Dict[str, str]], flags: int
    ) -> Tuple[List[str], Pattern[str]]:
        """
        Compile patterns into a single alternation regex.

        If a dict is given, build a pattern with **named** groups to preserve the
        provided names. If a list is given, synthesize names (k0, k1, ...).

        Args:
            patterns (Union[List[str], Dict[str, str]]): Patterns or mapping.
            flags (int): Regex flags.

        Returns:
            Tuple[List[str], Pattern[str]]: Names and compiled regex.
        """
        if isinstance(patterns, dict):
            names = list(patterns.keys())
            parts = [f"(?P<{name}>{pat})" for name, pat in patterns.items()]
        else:
            names = [f"k{i}" for i in range(len(patterns))]
            parts = [f"(?P<{n}>{pat})" for n, pat in zip(names, patterns)]

        combined = "|".join(parts) if parts else r"(?!x)x"  # never matches if empty
        compiled = re.compile(combined, flags)
        return names, compiled

    def _split_by_keywords(
        self, text: str
    ) -> Tuple[List[str], List[Tuple[int, int]], List[str]]:
        """
        Split ``text`` around matches of ``self.compiled``.

        Respects include_delimiters in {"before", "after", "both", "none"}.

        Args:
            text (str): The text to split.

        Returns:
            Tuple[List[str], List[Tuple[int, int]], List[str]]:
                (chunks, spans, names) where `chunks` are before size re-wrapping,
                spans are (start, end) tuples, and names are group names for each match.
        """

        def _append_chunk(acc: List[str], chunk: str) -> None:
            # Keep only non-empty (after strip) chunks here; final fallback to [""] is done by caller
            if chunk and chunk.strip():
                acc.append(chunk)

        chunks: List[str] = []
        spans: List[Tuple[int, int]] = []
        names: List[str] = []

        matches = list(self.compiled.finditer(text))
        last_idx = 0
        pending_prefix = ""  # used when include_delimiters is "after" or "both"

        for m in matches:
            start, end = m.span()
            match_txt = text[start:end]
            group_name = m.lastgroup or "unknown"

            spans.append((start, end))
            names.append(group_name)

            # Build the piece between last match end and this match start, prefixing any pending delimiter
            before_piece = pending_prefix + text[last_idx:start]
            pending_prefix = ""

            # Attach delimiter to the left side if requested
            if self.include_delimiters in ("before", "both"):
                before_piece += match_txt

            _append_chunk(chunks, before_piece)

            # If delimiter should be on the right, carry it forward to prefix next chunk
            if self.include_delimiters in ("after", "both"):
                pending_prefix = match_txt

            last_idx = end

        # Remainder after the last match (may contain pending_prefix)
        remainder = pending_prefix + text[last_idx:]
        _append_chunk(chunks, remainder)

        # If no non-empty chunks were appended, return a single empty chunk (tests expect this)
        if not chunks:
            return [""], spans, names

        # normalize whitespace trimming for each chunk
        chunks = [c.strip() for c in chunks if c and c.strip()]

        if not chunks:
            return [""], spans, names

        return chunks, spans, names

    @staticmethod
    def _soft_wrap(text: str, max_size: int) -> List[str]:
        """
        Greedy soft-wrap by whitespace to respect ``max_size``.

        - If ``len(text) <= max_size``: return ``[text]``.
        - Else: split on whitespace and rebuild lines greedily.
        - If a single token is longer than ``max_size``, it is hard-split.

        Args:
            text (str): Text to wrap.
            max_size (int): Maximum chunk size.

        Returns:
            List[str]: List of size-constrained chunks.
        """
        if max_size <= 0 or len(text) <= max_size:
            return [text] if text else []

        tokens = re.findall(r"\S+|\s+", text)
        out: List[str] = []
        buf = ""
        for tok in tokens:
            if len(buf) + len(tok) <= max_size:
                buf += tok
                continue
            if buf:
                out.append(buf)
                buf = ""
            # token alone is too big -> hard split
            while len(tok) > max_size:
                out.append(tok[:max_size])
                tok = tok[max_size:]
            buf = tok
        if buf:
            out.append(buf)
        return [c for c in (s.strip() for s in out) if c]

    @staticmethod
    def _count_by_name(names: Iterable[str]) -> Dict[str, int]:
        """
        Aggregate match counts by group name (k0/k1/... for list patterns, custom names for dict).

        Args:
            names (Iterable[str]): Group names.

        Returns:
            Dict[str, int]: Count of matches per group name.
        """
        counts: Dict[str, int] = {}
        for n in names:
            counts[n] = counts.get(n, 0) + 1
        return counts

    def _build_output(
        self,
        reader_output: ReaderOutput,
        chunks: List[str],
        chunk_ids: List[str],
        matches_meta: Dict[str, object],
    ) -> SplitterOutput:
        """
        Assemble a :class:`SplitterOutput` carrying over reader metadata.

        Args:
            reader_output (ReaderOutput): Input document and metadata.
            chunks (List[str]): Final list of chunks.
            chunk_ids (List[str]): Unique chunk IDs.
            matches_meta (Dict[str, object]): Keyword matches metadata.

        Returns:
            SplitterOutput: Populated output object.
        """
        return SplitterOutput(
            chunks=chunks,
            chunk_id=chunk_ids,
            document_name=reader_output.document_name,
            document_path=reader_output.document_path,
            document_id=reader_output.document_id,
            conversion_method=reader_output.conversion_method,
            reader_method=reader_output.reader_method,
            ocr_method=reader_output.ocr_method,
            split_method="keyword",
            split_params={
                "include_delimiters": self.include_delimiters,
                "flags": self.flags,
                "chunk_size": self.chunk_size,
                "pattern_names": self.pattern_names,
            },
            metadata={
                **(reader_output.metadata or {}),
                "keyword_matches": matches_meta,
            },
        )
