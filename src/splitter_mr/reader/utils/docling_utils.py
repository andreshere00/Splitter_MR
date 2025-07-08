import base64
import io
import re
import warnings
from pathlib import Path
from typing import Any, Callable, Dict, Tuple
from urllib.parse import urlencode, urljoin

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.types.doc import ImageRefMode
from openai import AzureOpenAI, OpenAI
from PIL.Image import Image

from ...model import BaseModel
from ...schema import DEFAULT_EXTRACTION_PROMPT, DEFAULT_IMAGE_CAPTION_PROMPT

# Helpers


def get_vlm_url_and_headers(client: Any) -> Tuple[str, dict]:
    """
    Build URL and headers for AzureOpenAI or OpenAI VLM endpoints using urllib.parse.

    Args:
        client: An instance of AzureOpenAI or OpenAI.

    Returns:
        Tuple containing the full endpoint URL and headers dictionary.

    Raises:
        ValueError: If required Azure parameters are missing or client type is unsupported.
    """
    headers = {"Authorization": f"Bearer {client.api_key}"}

    if isinstance(client, AzureOpenAI):
        endpoint = client._azure_endpoint
        deployment = client._azure_deployment
        version = client._api_version
        if not all([endpoint, deployment, version]):
            raise ValueError(
                "Missing Azure VLM config: endpoint, deployment, or api_version"
            )
        base = str(endpoint).rstrip("/") + "/"
        path = f"openai/deployments/{deployment}/chat/completions"
        url = urljoin(base, path)
        query = urlencode({"api-version": version})
        full_url = str(f"{url}?{query}")
        return full_url, headers

    if isinstance(client, OpenAI):
        base = "https://api.openai.com/"
        path = "v1/chat/completions"
        full_url = str(urljoin(base, path))
        return full_url, headers

    raise ValueError(f"Unsupported client type: {type(client)}")


# Pipelines


def page_image_pipeline(
    file_path: str | Path,
    model: BaseModel = None,
    prompt: str = DEFAULT_EXTRACTION_PROMPT,
    image_resolution: float = 1.0,
    show_base64_images: bool = False,
    page_placeholder: str = "<!-- page -->",
) -> str:
    """
    Processes a PDF by extracting each page as an image, then running a vision-language model on each page image.
    Returns extracted content for each page, separated by a page placeholder.

    Args:
        file_path (str): Path to the PDF file.
        model (BaseModel): Model instance used to extract text from each page image.
        prompt (str): Prompt/instruction for the model.
        image_resolution (float, optional): Scaling factor for output image resolution. Defaults to 1.0 (72 dpi).
        show_base64_images (bool): Whether to embed images in base64 format. If False, replaces images with placeholders.
        page_placeholder (str): Placeholder string for page breaks, e.g., '<!-- page -->'.

    Returns:
        output_md (str): Markdown-formatted string with each page's extracted content.

    Raises:
        ValueError: If a model is not provided and show_base64_images is False.
    """

    def image_to_base64(img: Image) -> str:
        """
        Helper to convert images to base64 formatted strings.

        Args:
            img (Image): an image from the PIL module.

        Returns:
            str: a base64 utf-8 decoded string.
        """
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    file_path = str(file_path)

    if model is None and show_base64_images is False:
        raise ValueError(
            "Either a model must be provided or show_base64_images must be True."
        )

    pipeline_options = PdfPipelineOptions(
        images_scale=image_resolution,
        generate_page_images=True,
        generate_picture_images=True,
    )
    doc_converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )
    conv_res = doc_converter.convert(file_path)
    output_md = ""
    for page_no, page in conv_res.document.pages.items():
        pil_img = page.image.pil_image
        img_base64 = image_to_base64(pil_img)
        if model:
            text = model.extract_text(prompt=prompt, file=img_base64)
            output_md += f"{page_placeholder}\n\n{text.strip()}\n\n"
        else:
            # Embed the image in markdown
            md_img = f"![Page {page_no}](data:image/png;base64,{img_base64})"
            output_md += f"{page_placeholder}\n\n{md_img}\n\n"
    return output_md


def vlm_pipeline(
    file_path: str | Path,
    model: BaseModel = None,
    prompt: str = DEFAULT_IMAGE_CAPTION_PROMPT,
    page_placeholder: str = "<!-- page -->",
    image_resolution: float = 1.0,
    image_placeholder: str = "<!-- image -->",
) -> str:
    """
    Processes a PDF using a remote Vision-Language Model (VLM) pipeline, returning the result as Markdown.

    Args:
        file_path (str): Path to the PDF file.
        model (Any): Model instance with a `get_client()` method and `model_name` attribute.
        prompt (str): Prompt for the VLM extraction.
        page_placeholder (str): Placeholder to indicate the start of a new page (e.g., '<!-- page -->').
        image_resolution (float, optional): Scaling factor for output image resolution. Defaults to 1.0 (72 dpi).
        image_placeholder (str): Placeholder string for images (when not embedding), e.g., '<!-- image -->'.

    Returns:
        md (str): Markdown-formatted extracted document.
    """
    file_path = str(file_path)

    if model is None:
        raise ValueError("A model must be provided for vlm_pipeline.")

    def describe_and_replace_base64_images(
        md: str, model: BaseModel, prompt: str, image_placeholder: str
    ) -> str:
        """
        Finds embedded base64 images in the markdown string, passes them to the model for a description,
        and replaces the image with a placeholder and the description. Logs a warning if processing fails.

        Args:
            md (str): The Markdown string.
            model (BaseModel): The model for image description.
            prompt (str): The prompt for the model.
            image_placeholder (str): The placeholder to use.

        Returns:
            md (str): Modified Markdown.
        """
        img_pattern = re.compile(
            r"!\[(.*?)\]\(data:image/(?:png|jpeg|jpg);base64,([A-Za-z0-9+/=\s]+)\)",
            re.DOTALL,
        )

        def replace_img(match):
            alt_text = match.group(1)
            img_b64 = match.group(2).replace("\n", "")  # Remove line breaks
            try:
                desc = model.extract_text(prompt=prompt, file=img_b64)
                if not desc or not desc.strip():
                    warnings.warn(
                        f"No description generated for image with alt text '{alt_text}'"
                    )
                    desc = "Image description not available."
            except Exception as e:
                warnings.warn(
                    f"Failed to process image with alt text '{alt_text}': {e}"
                )
                desc = f"Image extraction failed: {e}"
            return f"{image_placeholder}\n{desc.strip()}"

        return img_pattern.sub(replace_img, md)

    pipeline_options = PdfPipelineOptions(
        images_scale=image_resolution,
        generate_page_images=True,
        generate_picture_images=True,
    )
    doc_converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )
    conv_res = doc_converter.convert(file_path)
    md = conv_res.document.export_to_markdown(
        image_mode=ImageRefMode.EMBEDDED,
        page_break_placeholder=page_placeholder,
        image_placeholder=image_placeholder,
    )

    # Replace images with placeholder + description
    md = describe_and_replace_base64_images(md, model, prompt, image_placeholder)
    return md


def markdown_pipeline(
    file_path: str | Path,
    show_base64_images: bool = True,
    page_placeholder: str = "<!-- page -->",
    image_placeholder: str = "<!-- image -->",
    image_resolution: float = 1.0,
    ext: str = "pdf",
) -> str:
    """
    Processes a document using Docling's default Markdown extraction, with control over image embedding and placeholders.

    Args:
        file_path (str): Path to the document file.
        show_base64_images (bool): Whether to embed images in base64 format. If False, replaces images with placeholders.
        page_placeholder (str): Placeholder to indicate the start of a new page (e.g., '<!-- page -->').
        image_placeholder (str): Placeholder string for images (when not embedding), e.g., '<!-- image -->'.
        image_resolution (float, optional): Scaling factor for output image resolution. Defaults to 1.0 (72 dpi).
        ext (str): File extension.

    Returns:
        md (str): Markdown-formatted document with images handled per options.
    """

    file_path = str(file_path)

    if ext == "pdf":
        pipeline_options = PdfPipelineOptions(
            images_scale=image_resolution, generate_picture_images=True
        )
        doc_converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
            }
        )
    else:
        doc_converter = DocumentConverter()

    reader = doc_converter
    if show_base64_images:
        md = reader.convert(file_path).document.export_to_markdown(
            image_mode=ImageRefMode.EMBEDDED, page_break_placeholder=page_placeholder
        )
    else:
        md = reader.convert(file_path).document.export_to_markdown(
            image_mode=ImageRefMode.PLACEHOLDER,
            page_break_placeholder=page_placeholder,
            image_placeholder=image_placeholder,
        )
    return md


# Factory


class DoclingPipelineFactory:
    """
    Registry and orchestrator for Docling document pipelines.

    Allows registering new pipelines and dispatching calls to them in a unified way, using keyword arguments.
    Pipelines can have custom signatures, but must always accept 'file_path' as the first argument.
    """

    _registry: Dict[str, Callable] = {}

    @classmethod
    def register(cls, name: str, func: Callable[..., str]) -> None:
        """
        Registers a new pipeline function to the factory.

        Args:
            name (str): The unique name for this pipeline.
            func (Callable): The function implementing the pipeline. Should accept file_path and other kwargs.
        """
        cls._registry[name] = func

    @classmethod
    def get(cls, name: str) -> Callable[..., str]:
        """
        Retrieves a registered pipeline function by name.

        Args:
            name (str): Name of the registered pipeline.

        Returns:
            Callable[..., str]: The registered pipeline function.

        Raises:
            ValueError: If the pipeline name is not registered.
        """
        if name not in cls._registry:
            raise ValueError(f"Pipeline '{name}' not registered")
        return cls._registry[name]

    @classmethod
    def run(cls, pipeline_name: str, file_path: str, **kwargs) -> str:
        """
        Executes the pipeline by name with flexible arguments.

        Args:
            pipeline_name (str): Name of the registered pipeline.
            file_path (str): Path to the input document.
            **kwargs: Additional keyword arguments for the pipeline.

        Returns:
            str: Markdown-formatted extracted document or page text.

        Raises:
            ValueError: If the pipeline is not registered.
        """
        func = cls.get(pipeline_name)
        return func(file_path=file_path, **kwargs)


# Register pipelines
DoclingPipelineFactory.register("page_image", page_image_pipeline)
DoclingPipelineFactory.register("vlm", vlm_pipeline)
DoclingPipelineFactory.register("markdown", markdown_pipeline)
