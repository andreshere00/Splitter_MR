c = get_config()  # noqa: F821
c.Exporter.preprocessors = [
    "notebooks.truncate_output.HeadTailTruncateOutputPreprocessor"
]
c.HeadTailTruncateOutputPreprocessor.head_chars = 500
c.HeadTailTruncateOutputPreprocessor.tail_chars = 500
c.HeadTailTruncateOutputPreprocessor.ellipsis = "\n\n...\n\n"
