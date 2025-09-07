from nbconvert.preprocessors import Preprocessor
from traitlets import Integer, Unicode
from traitlets.config import Configurable


class HeadTailTruncateOutputPreprocessor(Preprocessor, Configurable):
    """
    Keep first N and last M lines of textual outputs.
    Handles:
      - stream (stdout/stderr): output['text'] is a STRING -> must stay a STRING
      - error: output['traceback'] is a LIST -> must stay a LIST
      - execute_result / display_data: data['text/plain'] usually STRING; keep type
    """

    head_chars = Integer(500, help="Caracteres al principio.").tag(config=True)
    tail_chars = Integer(500, help="Caracteres al final.").tag(config=True)
    ellipsis = Unicode("\n...\n", help="Marcador entre cabeza y cola.").tag(config=True)

    def _truncate_str(self, s: str):
        if s is None:
            return s, False
        keep = self.head_chars + self.tail_chars
        if len(s) <= keep:
            return s, False
        # Ya no usamos n_trunc
        return s[: self.head_chars] + self.ellipsis + s[-self.tail_chars :], True

    def _truncate_list_preserve_list(self, seq):
        """Para tracebacks (lista de líneas)."""
        if seq is None:
            return seq, False
        text = "".join(seq)
        new_text, truncated = self._truncate_str(text)
        if not truncated:
            return seq, False
        # Volver a lista de líneas con finales preservados
        return new_text.splitlines(keepends=True), True

    def preprocess_cell(self, cell, resources, index):
        for out in cell.get("outputs", []):
            ot = out.get("output_type")

            if ot == "stream":
                # Debe ser string
                new_text, truncated = self._truncate_str(out.get("text"))
                if truncated:
                    out["text"] = new_text
                    out.setdefault("metadata", {})["truncated_by_nbconvert"] = True

            elif ot == "error":
                # Debe ser lista
                new_tb, truncated = self._truncate_list_preserve_list(
                    out.get("traceback")
                )
                if truncated:
                    out["traceback"] = new_tb
                    out.setdefault("metadata", {})["truncated_by_nbconvert"] = True

            elif ot in ("execute_result", "display_data"):
                data = out.get("data", {})
                if "text/plain" in data:
                    # Forzar string (aunque viniera como lista)
                    tp = data.get("text/plain")
                    tp_str = "".join(tp) if isinstance(tp, list) else tp
                    new_tp, truncated = self._truncate_str(tp_str)
                    if truncated:
                        data["text/plain"] = new_tp
                        out.setdefault("metadata", {})["truncated_by_nbconvert"] = True

        return cell, resources
