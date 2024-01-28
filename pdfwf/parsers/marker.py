"""The marker PDF parser."""
from __future__ import annotations

from pdfwf.registry import register


@register()  # type: ignore[arg-type]
class MarkerParser:
    """Warmstart interface for the marker PDF parser.

    Initialization loads the marker models into memory and registers them in a
    global registry unique to the current process. This ensures that the models
    are only loaded once per worker process (i.e., we warmstart the models)
    """

    def __init__(self) -> None:
        """Initialize the marker parser."""
        from marker.models import load_all_models

        self.model_lst = load_all_models()

    def parse(self, pdf_path: str) -> tuple[str, dict[str, str]]:
        """Parse a PDF file and extract markdown."""
        from marker.convert import convert_single_pdf

        full_text, out_meta = convert_single_pdf(pdf_path, self.model_lst)

        return full_text, out_meta
