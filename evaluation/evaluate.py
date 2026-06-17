"""
evaluate.py — Evaluation framework for the ranking system.

Computes standard IR metrics:
- NDCG@K (Normalized Discounted Cumulative Gain)
- MAP (Mean Average Precision)
- P@K (Precision at K)

Also validates submission integrity:
- Honeypot rate in top 100
- Title distribution analysis
- Score monotonicity check
"""

import csv
import json
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))


def dcg_at_k(relevances, k):
    """Compute DCG@K from a list of relevance scores."""
    relevances = relevances[:k]
    return sum(
        rel / math.log2(i + 2)  # i+2 because log2(1)=0, we start from rank 1
        for i, rel in enumerate(relevances)
    )


def ndcg_at_k(relevances, k):
    """Compute NDCG@K."""
    dcg = dcg_at_k(relevances, k)
    ideal = dcg_at_k(sorted(relevances, reverse=True), k)
    return dcg / ideal if ideal > 0 else 0.0


def precision_at_k(relevances, k, threshold=1):
    """Compute P@K — fraction of top-K items with relevance >= threshold."""
    top_k = relevances[:k]
    if not top_k:
        return 0.0
    return sum(1 for r in top_k if r >= threshold) / len(top_k)


def mean_average_precision(relevances, threshold=1):
    """Compute MAP across all ranked items."""
    hits = 0
    sum_precision = 0.0
    for i, rel in enumerate(relevances):
        if rel >= threshold:
            hits += 1
            sum_precision += hits / (i + 1)
    return sum_precision / hits if hits > 0 else 0.0


def compute_composite_score(relevances):
    """
    Compute the official hackathon composite score:
    0.50 * NDCG@10 + 0.30 * NDCG@50 + 0.15 * MAP + 0.05 * P@10
    """
    ndcg10 = ndcg_at_k(relevances, 10)
    ndcg50 = ndcg_at_k(relevances, 50)
    map_score = mean_average_precision(relevances)
    p10 = precision_at_k(relevances, 10)

    composite = 0.50 * ndcg10 + 0.30 * ndcg50 + 0.15 * map_score + 0.05 * p10

    return {
        'NDCG@10': ndcg10,
        'NDCG@50': ndcg50,
        'MAP': map_score,
        'P@10': p10,
        'P@5': precision_at_k(relevances, 5),
        'composite': composite,
    }


def validate_submission_integrity(csv_path, candidates_path=None):
    """
    Validate submission CSV for common issues.

    Returns:
        dict with validation results
    """
    results = {
        'valid': True,
        'issues': [],
        'stats': {},
    }

    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
    except Exception as e:
        results['valid'] = False
        results['issues'].append(f"Cannot read CSV: {e}")
        return results

    # Check row count
    if len(rows) != 100:
        results['issues'].append(f"Expected 100 rows, got {len(rows)}")
        results['valid'] = False

    # Check ranks
    ranks = set()
    scores = []
    candidate_ids = set()

    for row in rows:
        rank = int(row.get('rank', 0))
        score = float(row.get('score', 0))
        cid = row.get('candidate_id', '')

        ranks.add(rank)
        scores.append((rank, score))
        candidate_ids.add(cid)

    # Missing ranks
    expected_ranks = set(range(1, 101))
    missing = expected_ranks - ranks
    if missing:
        results['issues'].append(f"Missing ranks: {sorted(missing)}")
        results['valid'] = False

    # Duplicate candidate_ids
    if len(candidate_ids) != len(rows):
        results['issues'].append("Duplicate candidate_ids found")
        results['valid'] = False

    # Score monotonicity
    scores.sort(key=lambda x: x[0])
    for i in range(len(scores) - 1):
        if scores[i][1] < scores[i+1][1]:
            results['issues'].append(
                f"Score not monotonically decreasing: rank {scores[i][0]} "
                f"({scores[i][1]}) < rank {scores[i+1][0]} ({scores[i+1][1]})"
            )
            results['valid'] = False
            break

    # Check reasoning variation
    reasonings = [row.get('reasoning', '') for row in rows]
    unique_reasonings = len(set(reasonings))
    if unique_reasonings < 90:
        results['issues'].append(
            f"Low reasoning variation: {unique_reasonings}/100 unique"
        )

    # Empty reasonings
    empty = sum(1 for r in reasonings if not r.strip())
    if empty > 0:
        results['issues'].append(f"{empty} empty reasoning fields")

    results['stats'] = {
        'total_rows': len(rows),
        'unique_candidate_ids': len(candidate_ids),
        'unique_reasonings': unique_reasonings,
        'score_range': (scores[0][1], scores[-1][1]) if scores else (0, 0),
        'empty_reasonings': empty,
    }

    return results


def analyze_submission(csv_path, candidates_path):
    """
    Detailed analysis of a submission against the candidate pool.
    Checks for honeypots, title distribution, etc.
    """
    # Load submission
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        submission = {row['candidate_id']: row for row in reader}

    # Load relevant candidates from the pool
    submission_ids = set(submission.keys())
    candidates = {}

    with open(candidates_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            cand = json.loads(line)
            if cand['candidate_id'] in submission_ids:
                candidates[cand['candidate_id']] = cand

    # Analyze titles
    title_counts = {}
    for cid, cand in candidates.items():
        title = cand['profile']['current_title']
        title_counts[title] = title_counts.get(title, 0) + 1

    # Check for honeypots
    from src.honeypot_detector import detect_honeypot
    honeypots_in_top100 = []
    for cid, cand in candidates.items():
        is_hp, reasons = detect_honeypot(cand)
        if is_hp:
            rank = int(submission[cid]['rank'])
            honeypots_in_top100.append({
                'candidate_id': cid,
                'rank': rank,
                'reasons': reasons,
            })

    # Location distribution
    location_counts = {}
    for cid, cand in candidates.items():
        loc = cand['profile'].get('country', 'Unknown')
        location_counts[loc] = location_counts.get(loc, 0) + 1

    print(f"\n{'='*60}")
    print(f"  Submission Analysis")
    print(f"{'='*60}")

    print(f"\n  Title Distribution (top 100):")
    for title, count in sorted(title_counts.items(), key=lambda x: -x[1]):
        print(f"    {title}: {count}")

    print(f"\n  Country Distribution (top 100):")
    for loc, count in sorted(location_counts.items(), key=lambda x: -x[1]):
        print(f"    {loc}: {count}")

    if honeypots_in_top100:
        print(f"\n  [!] HONEYPOTS FOUND IN TOP 100: {len(honeypots_in_top100)}")
        for hp in honeypots_in_top100:
            print(f"    Rank {hp['rank']}: {hp['candidate_id']} - {hp['reasons'][0]}")
    else:
        print(f"\n  [OK] No honeypots found in top 100")

    honeypot_rate = len(honeypots_in_top100) / 100.0
    print(f"\n  Honeypot rate: {honeypot_rate:.0%} "
          f"{'PASS' if honeypot_rate <= 0.10 else 'FAIL - DISQUALIFIED'}")

    return {
        'title_distribution': title_counts,
        'location_distribution': location_counts,
        'honeypots': honeypots_in_top100,
        'honeypot_rate': honeypot_rate,
    }


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate a ranking submission.")
    parser.add_argument('--submission', '-s', required=True, help='Path to submission CSV')
    parser.add_argument('--candidates', '-c', help='Path to candidates.jsonl (for analysis)')
    args = parser.parse_args()

    print("Validating submission format...")
    validation = validate_submission_integrity(args.submission)

    if validation['valid']:
        print("[OK] Submission format is valid")
    else:
        print("[FAIL] Submission format has issues:")
        for issue in validation['issues']:
            print(f"  - {issue}")

    print(f"\nStats: {validation['stats']}")

    if args.candidates:
        analyze_submission(args.submission, args.candidates)
