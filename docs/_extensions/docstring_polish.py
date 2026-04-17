"""Griffe extension that polishes numpy-style docstrings for mkdocstrings rendering.

Two transforms are applied at the source / parsed-section level so the rest of
mkdocstrings's pipeline (cross-references, inventories, default templates)
just works on the result.

1. Strip trailing ` (optional)` from parameter type lines. The project writes
   optional parameters as `name : Type (optional)`; griffe's numpy parser only
   strips `, optional`, so without this the annotation ends up parsed as a
   Python call `Type(optional)` which renders verbatim. The `Default` column
   already conveys optionality.

2. Split Raises entries of the form `name : ExceptionType` into a real `name`
   attribute plus a typed `annotation` expression. Griffe's numpy parser only
   recognises bare exception types (`ExceptionType`) — the `name :` prefix
   makes it leave the whole line as a raw string, which blocks cross-reference
   resolution (e.g. linking `ValueError` to python.org). After this transform
   the custom `raises.html.jinja` template can render the type via the
   standard expression pipeline and get autoref'd inventory links for free.
"""

from __future__ import annotations

import re

import griffe
from griffe._internal.docstrings.utils import parse_docstring_annotation


_OPTIONAL_RE = re.compile(r"^(?P<prefix>\s*\w[\w\d_]*\s*:\s*.+?)\s*\(optional\)\s*$", re.MULTILINE)


def _strip_optional(text: str | None) -> str | None:
    if not text or "(optional)" not in text:
        return text
    return _OPTIONAL_RE.sub(r"\g<prefix>", text)


def _split_raises(docstring: griffe.Docstring) -> None:
    for section in docstring.parsed:
        if section.kind != griffe.DocstringSectionKind.raises:
            continue
        for raise_obj in section.value:
            annot = raise_obj.annotation
            if not isinstance(annot, str) or " : " not in annot:
                continue
            name_str, type_str = annot.split(" : ", 1)
            raise_obj.name = name_str.strip()
            raise_obj.annotation = parse_docstring_annotation(type_str.strip(), docstring)


class DocstringPolish(griffe.Extension):
    """Project-specific numpy docstring cleanup for mkdocstrings."""

    def on_instance(
        self,
        *,
        obj: griffe.Object,
        **_: object,
    ) -> None:
        doc = obj.docstring
        if doc is None:
            return

        cleaned = _strip_optional(doc.value)
        if cleaned is not None and cleaned != doc.value:
            doc.value = cleaned
            doc.__dict__.pop("parsed", None)

        _split_raises(doc)
