#!/usr/bin/env python3
"""Verify the paragraph index against the downloaded government findings page."""
import json
import re
from pathlib import Path

from bs4 import BeautifulSoup


def main():
    root = Path(__file__).resolve().parent
    source = root / 'sources' / 'findings.html'
    soup = BeautifulSoup(source.read_text(encoding='utf-8'), 'html.parser')
    body = soup.select_one('.body-content')
    if body is None:
        raise ValueError('Government findings page has no publication body')
    sections = {}
    heading = None
    for element in body.find_all(['h3', 'h4', 'p']):
        text = re.sub(r'\s+', ' ', element.get_text(' ', strip=True)).strip()
        if element.name in ('h3', 'h4'):
            heading = text
            sections[heading] = []
        elif heading is not None:
            sections[heading].append(text)
    expected = json.loads((root / 'official_findings_sections.json').read_text())
    if sections != expected:
        raise ValueError('Findings page differs from the archived paragraph index. Review source changes before using the fixed paragraph map.')
    print(f'PASS: {len(sections)} findings sections match the archived paragraph index')


if __name__ == '__main__':
    main()
