#!/usr/bin/env python3
"""
nb2md.py — Convert Jupyter notebooks to clean Markdown.

Usage:
    python nb2md.py *.ipynb
"""
# Written with Claude (sonnet 4.6)

import argparse
import json
import re
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def strip_ansi(text: str) -> str:
    """Remove ANSI escape sequences from terminal output."""
    return re.sub(r"\x1b\[[0-9;]*m", "", text)


def clean_html(text: str) -> str:
    """
    Strip most HTML tags that Jupyter injects into text/html output,
    keeping the raw content. Also collapses excessive blank lines.
    """
    # Remove <style> blocks entirely
    text = re.sub(r"<style[^>]*>.*?</style>", "", text, flags=re.DOTALL)
    # Remove <script> blocks entirely
    text = re.sub(r"<script[^>]*>.*?</script>", "", text, flags=re.DOTALL)
    # Replace <br> / <br/> with newline
    text = re.sub(r"<br\s*/?>", "\n", text)
    # Replace common block elements with newlines
    text = re.sub(r"</(div|p|tr|thead|tbody|table)>", "\n", text, flags=re.IGNORECASE)
    # Replace <td> / <th> with a tab separator (for tables)
    text = re.sub(r"<t[dh][^>]*>", "\t", text, flags=re.IGNORECASE)
    # Strip all remaining tags
    text = re.sub(r"<[^>]+>", "", text)
    # Decode common HTML entities
    text = (text
            .replace("&amp;", "&")
            .replace("&lt;", "<")
            .replace("&gt;", ">")
            .replace("&nbsp;", " ")
            .replace("&quot;", '"')
            .replace("&#39;", "'"))
    # Collapse 3+ blank lines into 2
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def join_lines(lines: list[str]) -> str:
    return "".join(lines)


# ---------------------------------------------------------------------------
# Output data handlers
# ---------------------------------------------------------------------------

def handle_image(mime: str, data: str) -> str:
    """Return an inline HTML <img> tag with base64-encoded image data."""
    if mime == "image/svg+xml":
        # SVG can be embedded directly as raw HTML (no base64 needed)
        return f"{data}\n"
    # For raster formats, use a data URI
    return f'<img src="data:{mime};base64,{data}" alt="Output figure" />\n'


def render_output(output: dict) -> str:
    """Convert a single cell output to Markdown text."""
    output_type = output.get("output_type", "")
    parts = []

    if output_type in ("display_data", "execute_result"):
        data = output.get("data", {})

        # Prefer images over text
        for mime in ("image/png", "image/jpeg", "image/svg+xml", "image/gif"):
            if mime in data:
                raw = data[mime]
                # SVG / data may be a list of strings
                if isinstance(raw, list):
                    raw = join_lines(raw)
                parts.append(handle_image(mime, raw))
                return "".join(parts)  # image found, skip text representations

        # text/html — strip HTML and show as plain text block
        if "text/html" in data:
            html = join_lines(data["text/html"]) if isinstance(data["text/html"], list) else data["text/html"]
            cleaned = clean_html(html)
            if cleaned:
                parts.append(f"```\n{cleaned}\n```\n")
            return "".join(parts)

        # text/plain fallback
        if "text/plain" in data:
            text = join_lines(data["text/plain"]) if isinstance(data["text/plain"], list) else data["text/plain"]
            text = strip_ansi(text).rstrip()
            if text:
                parts.append(f"```\n{text}\n```\n")

    elif output_type == "stream":
        text = join_lines(output.get("text", []))
        text = strip_ansi(text).rstrip()
        if text:
            name = output.get("name", "stdout")
            label = "" if name == "stdout" else f"  <!-- {name} -->"
            parts.append(f"```{label}\n{text}\n```\n")

    elif output_type == "error":
        ename = output.get("ename", "Error")
        evalue = output.get("evalue", "")
        traceback = output.get("traceback", [])
        tb_clean = strip_ansi("\n".join(traceback))
        parts.append(f"```\n{ename}: {evalue}\n{tb_clean}\n```\n")

    return "".join(parts)


# ---------------------------------------------------------------------------
# Cell handlers
# ---------------------------------------------------------------------------

def render_code_cell(cell: dict) -> str:
    source = join_lines(cell.get("source", []))
    outputs = cell.get("outputs", [])

    parts = []
    if source.strip():
        parts.append(f"```python\n{source}\n```\n")

    for output in outputs:
        rendered = render_output(output)
        if rendered:
            parts.append(rendered)

    return "\n".join(parts)


def render_markdown_cell(cell: dict) -> str:
    source = join_lines(cell.get("source", []))
    return source.rstrip() + "\n"


def render_raw_cell(cell: dict) -> str:
    source = join_lines(cell.get("source", []))
    if source.strip():
        return f"```\n{source.rstrip()}\n```\n"
    return ""


# ---------------------------------------------------------------------------
# Main converter
# ---------------------------------------------------------------------------

def convert(nb_path: Path) -> str:
    nb = json.loads(nb_path.read_text(encoding="utf-8"))

    cells = nb.get("cells", [])
    sections = []

    for cell in cells:
        cell_type = cell.get("cell_type", "")
        if cell_type == "code":
            rendered = render_code_cell(cell)
        elif cell_type == "markdown":
            rendered = render_markdown_cell(cell)
        elif cell_type == "raw":
            rendered = render_raw_cell(cell)
        else:
            continue

        if rendered.strip():
            sections.append(rendered)

    return "\n\n".join(sections) + "\n"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Convert a Jupyter notebook to Markdown.")
    parser.add_argument("notebooks", type=Path, help="Path to the .ipynb file", nargs="*")
    args = parser.parse_args()

    for notebook in args.notebooks:

        if not notebook.exists():
            print(f"Error: {notebook} not found.", file=sys.stderr)
            continue

        md = convert(notebook)

        notebook.with_suffix(".md").write_text(md, encoding="utf-8")
        print(f"Written to {notebook.with_suffix('.md')}", file=sys.stderr)


if __name__ == "__main__":
    main()
