DEFAULT_EXTRACTION_PROMPT: str = (
    "Extract all the elements (text, formulas, tables, images, etc.) detected in the page in markdown format, orderly. Return ONLY the extracted content, with no previous comments or placeholders."
)

DEFAULT_IMAGE_CAPTION_PROMPT: str = (
    "Provide a caption describing the following resource. Return the output with the following format: *Caption: <A brief description>*."
)
