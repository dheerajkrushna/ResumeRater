"""
loader.py — Candidate data loading from JSONL files.

Handles both plain .jsonl and gzipped .jsonl.gz files.
Provides streaming iteration to keep memory usage low.
"""

import json
import gzip
from pathlib import Path


def load_candidates(file_path, limit=None):
    """
    Load candidates from a JSONL or JSONL.GZ file.

    Args:
        file_path: Path to the candidates file (.jsonl or .jsonl.gz)
        limit: Optional maximum number of candidates to load (for testing)

    Yields:
        dict: Parsed candidate record
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"Candidate file not found: {file_path}")

    opener = gzip.open if path.suffix == '.gz' else open
    mode = 'rt' if path.suffix == '.gz' else 'r'

    count = 0
    with opener(path, mode, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                candidate = json.loads(line)
                yield candidate
                count += 1
                if limit and count >= limit:
                    break
            except json.JSONDecodeError as e:
                print(f"Warning: Skipping malformed JSON line: {e}")
                continue

    # If it's a .json file (array format), handle that too
    if path.suffix == '.json' and count == 0:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if isinstance(data, list):
                for i, candidate in enumerate(data):
                    yield candidate
                    if limit and i + 1 >= limit:
                        break


def load_all_candidates(file_path, limit=None):
    """
    Load all candidates into memory as a list.

    Args:
        file_path: Path to the candidates file
        limit: Optional maximum number of candidates

    Returns:
        list[dict]: List of candidate records
    """
    return list(load_candidates(file_path, limit=limit))


def extract_candidate_text(candidate):
    """
    Extract a single text document from a candidate's profile for TF-IDF vectorization.

    Concatenates: headline, summary, all career descriptions, skill names,
    certifications, and education fields into one searchable text block.

    Args:
        candidate: dict — a single candidate record

    Returns:
        str: concatenated text representation of the candidate
    """
    parts = []
    profile = candidate.get('profile', {})

    # Profile text
    parts.append(profile.get('headline', ''))
    parts.append(profile.get('summary', ''))
    parts.append(profile.get('current_title', ''))
    parts.append(profile.get('current_industry', ''))

    # Career history descriptions and titles
    for job in candidate.get('career_history', []):
        parts.append(job.get('title', ''))
        parts.append(job.get('company', ''))
        parts.append(job.get('industry', ''))
        parts.append(job.get('description', ''))

    # Skills — include name and proficiency
    for skill in candidate.get('skills', []):
        name = skill.get('name', '')
        prof = skill.get('proficiency', '')
        parts.append(f"{name} {prof}")

    # Certifications
    for cert in candidate.get('certifications', []):
        parts.append(cert.get('name', ''))
        parts.append(cert.get('issuer', ''))

    # Education
    for edu in candidate.get('education', []):
        parts.append(edu.get('institution', ''))
        parts.append(edu.get('degree', ''))
        parts.append(edu.get('field_of_study', ''))

    return ' '.join(filter(None, parts)).lower()
