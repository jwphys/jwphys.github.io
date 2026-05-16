#!/usr/bin/env python3
"""
Fetch publications from Google Scholar and write to _data/publications.yml.
Uses the scholarly library (or manual scraping via serpapi as fallback).
"""
import os
import yaml
import re
from pathlib import Path

SCHOLAR_ID = os.environ.get("SCHOLAR_ID", "")

PUBLICATIONS_FILE = Path(__file__).resolve().parent.parent.parent / "_data" / "publications.yml"

def main():
    if not SCHOLAR_ID:
        print("No SCHOLAR_ID set. Using existing publications file.")
        return

    try:
        from scholarly import scholarly

        author = scholarly.search_author_id(SCHOLAR_ID)
        scholarly.fill(author, sections=["publications"])

        pubs = []
        for pub in author.get("publications", []):
            scholarly.fill(pub, sections=["bib"])
            bib = pub.get("bib", {})
            entry = {
                "year": bib.get("pub_year", ""),
                "title": bib.get("title", ""),
                "authors": bib.get("author", ""),
                "journal": bib.get("journal", ""),
                "doi": extract_doi(bib.get("doi", "")),
            }
            pubs.append(entry)

        # Sort by year descending
        pubs.sort(key=lambda x: x.get("year", "0"), reverse=True)

        with open(PUBLICATIONS_FILE, "w", encoding="utf-8") as f:
            yaml.dump(pubs, f, allow_unicode=True, sort_keys=False)

        print(f"Updated {len(pubs)} publications.")

    except ImportError:
        print("scholarly not installed. Install with: pip install scholarly")
    except Exception as e:
        print(f"Error fetching publications: {e}")


def extract_doi(text):
    if not text:
        return ""
    # Try to extract DOI from various formats
    match = re.search(r"10\.\d{4,}/[-._;()/:\w]+", str(text))
    return match.group(0) if match else text


if __name__ == "__main__":
    main()
