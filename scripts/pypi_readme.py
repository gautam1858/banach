#!/usr/bin/env python3
"""Print README.md with relative links made absolute, for the PyPI project page.

PyPI renders the README outside the repository, so `brand/hero-card.svg` or `docs/VALIDATION.md`
would be broken images and dead links there. Images point at raw.githubusercontent.com and other
files at their github.com page; anchors (#...) and full URLs are left alone.
"""
import re
import sys

REPO = "gautam1858/banach"
RAW = f"https://raw.githubusercontent.com/{REPO}/main/"
BLOB = f"https://github.com/{REPO}/blob/main/"


def absolute(path, base):
    return path if re.match(r"^(https?:|#|mailto:)", path) else base + path.lstrip("./")


text = open(sys.argv[1], encoding="utf-8").read()
text = re.sub(r'src="([^"]+)"', lambda m: f'src="{absolute(m[1], RAW)}"', text)
text = re.sub(r'href="([^"]+)"', lambda m: f'href="{absolute(m[1], BLOB)}"', text)
text = re.sub(r"\]\(([^)\s]+)\)", lambda m: f"]({absolute(m[1], BLOB)})", text)
sys.stdout.write(text)
