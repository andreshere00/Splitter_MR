# ipynb2md.py
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Iterable, Optional, Tuple

import nbformat

# nbconvert 7 / 6 compatible imports
try:
    from nbconvert.exporters import MarkdownExporter  # nbconvert >= 7
except Exception:  # pragma: no cover
    from nbconvert import MarkdownExporter  # nbconvert <= 6
try:
    from nbconvert.writers import FilesWriter
except Exception:  # pragma: no cover
    from nbconvert.writers.files import FilesWriter


@dataclass
class Settings:
    input_dir: Path
    output_dir: Path
    template_dir: Path
    preprocessor_path: str  # "pkg.mod:ClassName" or "path/to/file.py:ClassName"
    head_chars: Optional[int] = None
    tail_chars: Optional[int] = None
    ellipsis: Optional[str] = None
    strip_prefix: Optional[str] = None  # exact leading path to drop
    cut_until: Optional[str] = None  # anchor to cut up to (inclusive)
    dry_run: bool = False
    verbose: bool = False
    require_cut_until: bool = False
    no_sandbox: bool = False


def iter_ipynb_files(root: Path) -> Iterable[Path]:
    """Yield all .ipynb files under root, recursively, ignoring hidden dirs and checkpoints."""
    for p in root.rglob("*.ipynb"):
        if ".ipynb_checkpoints" in p.parts:
            continue
        if any(part.startswith(".") for part in p.parts):
            continue
        yield p


def _posix(pathlike: Path | str) -> PurePosixPath:
    return PurePosixPath(str(pathlike).replace("\\", "/"))


def compute_rel_out(
    ipynb_path: Path,
    src_root: Path,
    strip_prefix: Optional[str],
    cut_until: Optional[str],
) -> tuple[PurePosixPath, bool]:
    """
    Compute relative path under output_dir.

    Returns (relpath, anchor_used)

    1) If strip_prefix is set and relpath *starts with* it (exact path-part match),
       drop that leading prefix. (anchor_used = True)
    2) Else if cut_until is set and its path sequence appears anywhere in relpath,
       drop everything up to and including its first occurrence. (anchor_used = True)
    3) Else keep the full relpath. (anchor_used = False)
    """
    rel = _posix(
        ipynb_path.relative_to(src_root)
    )  # e.g. docs/examples/schema/foo.ipynb

    # (1) exact leading prefix drop
    if strip_prefix:
        pref = _posix(strip_prefix)
        if rel.parts[: len(pref.parts)] == pref.parts:
            return (PurePosixPath(*rel.parts[len(pref.parts) :]), True)

    # (2) anchor cut anywhere (inclusive)
    if cut_until:
        anchor = _posix(cut_until)
        parts = rel.parts
        for i in range(0, len(parts) - len(anchor.parts) + 1):
            if tuple(parts[i : i + len(anchor.parts)]) == anchor.parts:
                return (PurePosixPath(*parts[i + len(anchor.parts) :]), True)

    # (3) unchanged
    return (rel, False)


def load_preprocessor(preproc_path: str):
    """
    Load a preprocessor class from:
      - module form:   "pkg.mod:ClassName"
      - file path form:"path/to/file.py:ClassName"
    """
    import importlib.util

    try:
        module_name, class_name = preproc_path.split(":")
    except ValueError:
        raise ValueError(
            f'Invalid --preprocessor "{preproc_path}". '
            'Expected "pkg.mod:ClassName" or "path/to/file.py:ClassName".'
        )

    if module_name.endswith(".py"):
        p = Path(module_name).resolve()
        if not p.exists():
            raise FileNotFoundError(p)
        spec = importlib.util.spec_from_file_location(p.stem, p)
        mod = importlib.util.module_from_spec(spec)
        assert spec and spec.loader
        spec.loader.exec_module(mod)
    else:
        mod = importlib.import_module(module_name)

    cls = getattr(mod, class_name, None)
    if cls is None:
        raise ImportError(f'Class "{class_name}" not found in "{module_name}"')
    return cls


def build_exporter(settings: Settings) -> MarkdownExporter:
    """
    Build a MarkdownExporter configured with:
      - template_dir (expects index.md.j2)
      - registered preprocessor
      - optional traitlet overrides (head/tail/ellipsis)
    """
    exporter = MarkdownExporter(
        template_file="index.md.j2",
        template_path=[str(settings.template_dir)],
    )

    PreprocClass = load_preprocessor(settings.preprocessor_path)
    exporter.register_preprocessor(PreprocClass, enabled=True)

    # Configure traitlets on the preprocessor via exporter's Config
    cfg = exporter.config
    sect = PreprocClass.__name__
    overrides = {}
    if settings.head_chars is not None:
        overrides["head_chars"] = settings.head_chars
    if settings.tail_chars is not None:
        overrides["tail_chars"] = settings.tail_chars
    if settings.ellipsis is not None:
        overrides["ellipsis"] = settings.ellipsis
    if overrides:
        cfg.merge({sect: overrides})

    return exporter


def write_md(md_path: Path, body: str) -> Tuple[bool, bool]:
    """
    Write Markdown to md_path.
    Returns (created, changed).
    """
    md_path.parent.mkdir(parents=True, exist_ok=True)
    if md_path.exists():
        old = md_path.read_text(encoding="utf-8")
        if old == body:
            return (False, False)
    md_path.write_text(body, encoding="utf-8")
    return (not md_path.exists(), True)  # created: False if it existed; changed: True


def convert_one(
    ipynb_path: Path,
    dst_root: Path,
    src_root: Path,
    exporter: MarkdownExporter,
    strip_prefix: Optional[str],
    cut_until: Optional[str],
    dry_run: bool = False,
    verbose: bool = False,
    require_cut_until: bool = False,
) -> Optional[Path]:
    rel_out, anchor_used = compute_rel_out(
        ipynb_path, src_root, strip_prefix, cut_until
    )  # e.g. schema/foo.ipynb

    if require_cut_until and not anchor_used:
        if verbose:
            print(
                f"[SKIP] {ipynb_path.relative_to(src_root)}  (anchor '{cut_until}' not found)"
            )
        return None

    out_dir = dst_root.joinpath(*rel_out.parent.parts)  # docs/examples/schema
    stem = ipynb_path.stem
    md_path = out_dir / f"{stem}.md"

    # --- sandbox: never allow writes outside output_dir ---
    dst_root_abs = dst_root.resolve()
    md_path_abs = md_path.resolve().parent  # check parent dir
    if not str(md_path_abs).startswith(str(dst_root_abs)):
        raise RuntimeError(
            f"Refusing to write outside output dir:\n  md: {md_path}\n  out: {dst_root}"
        )

    if verbose:
        try:
            print(
                f"[MAP] {ipynb_path.relative_to(src_root)}  ->  {md_path.relative_to(dst_root)}"
            )
        except Exception:
            print(f"[MAP] {ipynb_path}  ->  {md_path}")

    if dry_run:
        return md_path

    resources = {
        "unique_key": stem,
        "output_files_dir": f"{stem}_files",
        "metadata": {"path": str(out_dir)},
    }

    nb = nbformat.read(str(ipynb_path), as_version=4)
    body, res = exporter.from_notebook_node(nb, resources=resources)

    created, changed = False, True
    out_dir.mkdir(parents=True, exist_ok=True)
    if md_path.exists():
        old = md_path.read_text(encoding="utf-8")
        if old == body:
            changed = False
        else:
            md_path.write_text(body, encoding="utf-8")
    else:
        md_path.write_text(body, encoding="utf-8")
        created = True

    writer = FilesWriter()
    writer.write(
        output=body, resources=res, notebook_name=stem, build_directory=str(out_dir)
    )

    if verbose:
        status = "created" if created else ("updated" if changed else "unchanged")
        print(f"[WRITE] {md_path.relative_to(dst_root)}  ({status})")

    return md_path


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Recursively convert .ipynb under --input into Markdown under --output, preserving hierarchy."
    )
    p.add_argument(
        "--input", required=True, help="Input directory containing notebooks (root)."
    )
    p.add_argument(
        "--output", required=True, help="Output directory for Markdown (root)."
    )
    p.add_argument(
        "--template-dir",
        default="notebooks/templates/markdown_fenced",
        help="Directory containing index.md.j2 (default: notebooks/templates/markdown_fenced)",
    )
    p.add_argument(
        "--preprocessor",
        default="notebooks/truncate_output.py:HeadTailTruncateOutputPreprocessor",
        help='Preprocessor as "pkg.mod:ClassName" or "path/to/file.py:ClassName".',
    )
    p.add_argument(
        "--head-chars", type=int, default=None, help="Override preprocessor head_chars."
    )
    p.add_argument(
        "--tail-chars", type=int, default=None, help="Override preprocessor tail_chars."
    )
    p.add_argument(
        "--ellipsis",
        default=None,
        help=r"Override preprocessor ellipsis, e.g. '\n...\n'.",
    )
    p.add_argument(
        "--strip-prefix",
        default=None,
        help="If the notebook relpath starts with this prefix, strip it (exact leading match).",
    )
    p.add_argument(
        "--cut-until",
        default=None,
        help="Drop everything up to and including the first occurrence of this anchor in the relpath.",
    )
    p.add_argument(
        "--require-cut-until",
        action="store_true",
        help="If set, only convert notebooks where --cut-until matched; otherwise skip.",
    )
    p.add_argument(
        "--no-sandbox",
        action="store_true",
        help="Disable the safety check that prevents writing outside --output (not recommended).",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned outputs without writing files.",
    )
    p.add_argument(
        "--verbose", action="store_true", help="Verbose mapping + write status."
    )
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    settings = Settings(
        input_dir=Path(args.input).resolve(),
        output_dir=Path(args.output).resolve(),
        template_dir=Path(args.template_dir).resolve(),
        preprocessor_path=args.preprocessor,
        head_chars=args.head_chars,
        tail_chars=args.tail_chars,
        ellipsis=args.ellipsis,
        strip_prefix=args.strip_prefix,
        cut_until=args.cut_until,
        dry_run=args.dry_run,
        verbose=args.verbose,
        require_cut_until=args.require_cut_until,
        no_sandbox=args.no_sandbox,
    )

    if not settings.input_dir.exists():
        print(f"[ERROR] Input directory not found: {settings.input_dir}")
        return 2
    if not settings.template_dir.exists():
        print(f"[ERROR] Template directory not found: {settings.template_dir}")
        return 2

    exporter = build_exporter(settings)

    ipynbs = sorted(iter_ipynb_files(settings.input_dir))
    if not ipynbs:
        print(f"[WARN] No .ipynb files found under {settings.input_dir}")
        return 0

    if settings.verbose:
        print(f"[INFO] Input root:  {settings.input_dir}")
        print(f"[INFO] Output root: {settings.output_dir}")
        print(f"[INFO] Template:    {settings.template_dir}")
        print(f"[INFO] Preproc:     {settings.preprocessor_path}")
        print(
            f"[INFO] strip_prefix:{settings.strip_prefix!r}  cut_until:{settings.cut_until!r}"
        )

    converted = []
    for nb_path in ipynbs:
        md_path = convert_one(
            ipynb_path=nb_path,
            dst_root=settings.output_dir,
            src_root=settings.input_dir,
            exporter=exporter,
            strip_prefix=settings.strip_prefix,
            cut_until=settings.cut_until,
            dry_run=settings.dry_run,
            verbose=settings.verbose,
            require_cut_until=settings.require_cut_until,
        )
        if md_path is not None:
            converted.append(md_path)

    print(
        f"{'Planned' if settings.dry_run else 'Converted'} {len(converted)} notebooks "
        f"from {settings.input_dir} to {settings.output_dir}."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
