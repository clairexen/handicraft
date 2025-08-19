from lxml import etree
import mwparserfromhell as mwp
import html, sys, bz2
from urllib.parse import quote
from pathlib import Path

NSANY = "{*}"  # namespace wildcard

def open_maybe_compressed(path):
    return bz2.open(path, "rb") if path.endswith(".bz2") else open(path, "rb")

def is_redirect(wikitext: str) -> bool:
    return (wikitext or "").lstrip().upper().startswith("#REDIRECT")

def to_plaintext(wikitext: str) -> str:
    """Best-effort plain text from MediaWiki markup."""
    if not wikitext:
        return ""
    code = mwp.parse(wikitext)
    text = code.strip_code(normalize=True, collapse=True)
    text = html.unescape(text)
    # compact trailing spaces and drop empty-only lines
    return "\n".join(line.rstrip() for line in text.strip().splitlines())

def ascii_with_unicode_escapes(s: str) -> str:
    """Keep ASCII (0x00–0x7F) as-is; escape others as \\uXXXX / \\UXXXXXXXX."""
    out = []
    for ch in s:
        cp = ord(ch)
        if cp < 0x80:
            out.append(ch)
        elif cp <= 0xFFFF:
            out.append(f"\\u{cp:04X}")
        else:
            out.append(f"\\U{cp:08X}")
    return "".join(out)

def wiki_url_from_title(title: str) -> str:
    # Simple English Wikipedia URL, ASCII-only via percent-encoding.
    # MediaWiki spaces → underscores; then quote to percent-encode.
    path = quote(title.replace(" ", "_"), safe=":/()!$&'*,;=+@-._~")  # keep common URL-safe chars
    return f"https://simple.wikipedia.org/wiki/{path}"

def iter_pages(path):
    ctx = etree.iterparse(open_maybe_compressed(path),
                          events=("end",),
                          tag=(NSANY + "page",))
    for _, page in ctx:
        try:
            # use wildcard in .find() as well
            ns_el   = page.find(".//" + NSANY + "ns")
            if ns_el is not None and ns_el.text != "0":
                continue

            title_el = page.find(".//" + NSANY + "title")
            text_el  = page.find(".//" + NSANY + "revision/" + NSANY + "text")

            title = (title_el.text or "") if title_el is not None else ""
            wikitext = text_el.text if text_el is not None else ""
            if not title or not wikitext or is_redirect(wikitext):
                continue
            yield title, wikitext
        finally:
            page.clear()
            while page.getprevious() is not None:
                del page.getparent()[0]

def dump_text(src_path, out_path1, out_path2, ratio):
    count = 0
    with open(out_path1, "w", encoding="utf-8", newline="\n") as out1:
        with open(out_path2, "w", encoding="utf-8", newline="\n") as out2:
            for count, (title, wikitext) in enumerate(iter_pages(src_path), 1):
                out = out1 if count % ratio else out2
                plain = to_plaintext(wikitext)
                if not plain.strip():
                    continue
                url = wiki_url_from_title(title)
                body = ascii_with_unicode_escapes(plain)
                out.write("REM '''\n")
                out.write(url + "\n\n")
                out.write(body.rstrip() + "\n")
                out.write("'''\n\n")
                if count % 10_000 == 0:
                    print(f"{count} pages...", file=sys.stderr)
    print(f"Done. Wrote {count} pages.", file=sys.stderr)

datapath = Path(f"datasets/simplewiki-plain")
datapath.mkdir(parents=True, exist_ok=True)

dump_text("datasrc/simplewiki-20250720-pages-articles.xml.bz2",
        "datasets/simplewiki-plain/simplewiki-train.asc",
        "datasets/simplewiki-plain/simplewiki-test.asc", 20)
