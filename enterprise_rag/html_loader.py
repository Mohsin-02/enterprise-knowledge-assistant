"""Parse local UTF-8 HTML with Python's standard-library HTMLParser."""

from html.parser import HTMLParser
import re

from .loaders import DocumentReadError, ParsedBlock, clean_text


BLOCK_TAGS = {"p", "li", "pre", "blockquote", "tr", "div", "section", "article", "main",
              "table", "ul", "ol", "h1", "h2", "h3", "h4", "h5", "h6"}
IGNORED_TAGS = {"head", "script", "style", "noscript", "template", "nav", "footer"}
VOID_TAGS = {"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta",
             "param", "source", "track", "wbr"}


class LocalHTMLParser(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.stack: list[dict] = []
        self.parts: list[str] = []
        self.blocks: list[ParsedBlock] = []
        self.headings: dict[int, tuple[str, str | None]] = {}

    def _ignored(self) -> bool:
        return bool(self.stack and self.stack[-1]["ignored"])

    def _flush(self) -> None:
        text = clean_text("".join(self.parts))
        self.parts.clear()
        if not text:
            return
        owner = next((frame for frame in reversed(self.stack) if frame["tag"] in BLOCK_TAGS), {})
        tag = owner.get("tag", "body")
        anchor = next((frame["id"] for frame in reversed(self.stack) if frame["id"]), None)
        if re.fullmatch(r"h[1-6]", tag):
            level = int(tag[1])
            self.headings = {key: value for key, value in self.headings.items() if key < level}
            self.headings[level] = (text, anchor)
        if anchor is None:
            anchor = next((item[1] for item in reversed(list(self.headings.values())) if item[1]), None)
        self.blocks.append(ParsedBlock(None, text, {
            "format": "html", "kind": "html_block", "block": len(self.blocks) + 1,
            "tag": tag, "anchor": anchor,
            "section": [item[0] for item in self.headings.values()],
        }))

    def handle_starttag(self, tag, attrs):
        attributes = dict(attrs)
        ignored = (self._ignored() or tag in IGNORED_TAGS or "hidden" in attributes
                   or (attributes.get("aria-hidden") or "").lower() == "true"
                   or bool(re.search(r"display\s*:\s*none\b|visibility\s*:\s*hidden\b",
                                     attributes.get("style") or "", flags=re.I)))
        if tag in VOID_TAGS:
            if not ignored and tag in {"br", "hr"}:
                self.parts.append("\n")
            return
        if tag in BLOCK_TAGS and not self._ignored():
            self._flush()
        if tag in {"td", "th"} and self.parts and not ignored:
            self.parts.append(" | ")
        self.stack.append({"tag": tag, "id": attributes.get("id"), "ignored": ignored})

    def handle_endtag(self, tag):
        index = next((i for i in range(len(self.stack) - 1, -1, -1)
                      if self.stack[i]["tag"] == tag), None)
        if index is None:
            return
        if tag in BLOCK_TAGS and not self._ignored():
            self._flush()
        del self.stack[index:]

    def handle_data(self, data):
        if not self._ignored():
            self.parts.append(data)


def load_html(data: bytes) -> tuple[list[ParsedBlock], list[str]]:
    try:
        markup = data.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise DocumentReadError("Save this HTML file as UTF-8 and try again.") from exc
    parser = LocalHTMLParser()
    parser.feed(markup)
    parser.close()
    parser._flush()
    return parser.blocks, []
