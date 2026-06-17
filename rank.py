#!/usr/bin/env python3
"""
rank.py — Entry point for the Intelligent Candidate Discovery & Ranking System.

Usage:
    python rank.py --candidates ./candidates.jsonl --out ./submission.csv

Produces a CSV with the top 100 candidates ranked by composite fit score.
Runs in <5 minutes on CPU with <16GB RAM, no network access required.
"""

import argparse
import csv
import sys
import time
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.loader import load_all_candidates
from src.semantic_scorer import SemanticScorer
from src.scorer import score_candidate
from src.reasoning_generator import generate_reasoning


def main():
    parser = argparse.ArgumentParser(
        description="Rank candidates for the Senior AI Engineer role."
    )
    parser.add_argument(
        '--candidates', '-c',
        required=True,
        help='Path to candidates.jsonl or candidates.jsonl.gz'
    )
    parser.add_argument(
        '--out', '-o',
        required=True,
        help='Path for the output CSV submission file'
    )
    parser.add_argument(
        '--top-n',
        type=int,
        default=100,
        help='Number of top candidates to include (default: 100)'
    )
    parser.add_argument(
        '--limit',
        type=int,
        default=None,
        help='Limit number of candidates to process (for testing)'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Print detailed progress information'
    )
    args = parser.parse_args()

    t_start = time.time()

    # ---------------------------------------------------------------
    # Step 1: Load all candidates
    # ---------------------------------------------------------------
    if args.verbose:
        print(f"[1/5] Loading candidates from {args.candidates}...")

    candidates = load_all_candidates(args.candidates, limit=args.limit)

    t_load = time.time()
    if args.verbose:
        print(f"       Loaded {len(candidates)} candidates in {t_load - t_start:.1f}s")

    # ---------------------------------------------------------------
    # Step 2: Compute semantic (TF-IDF) scores
    # ---------------------------------------------------------------
    if args.verbose:
        print("[2/5] Computing TF-IDF semantic scores...")

    semantic_scorer = SemanticScorer()
    semantic_scores = semantic_scorer.fit_and_score(candidates)

    t_semantic = time.time()
    if args.verbose:
        print(f"       Semantic scoring done in {t_semantic - t_load:.1f}s")

    # ---------------------------------------------------------------
    # Step 3: Score all candidates (hybrid: rule + semantic + behavioral)
    # ---------------------------------------------------------------
    if args.verbose:
        print("[3/5] Scoring all candidates...")

    results = []
    disqualified = 0
    honeypots_found = 0

    for cand in candidates:
        result = score_candidate(cand, semantic_scores=semantic_scores)

        if result['disqualified']:
            disqualified += 1
            if 'Honeypot' in result.get('disqualify_reason', ''):
                honeypots_found += 1
            continue

        if result['score'] > 0:
            results.append(result)

    t_score = time.time()
    if args.verbose:
        print(f"       Scored {len(results)} qualified candidates in {t_score - t_semantic:.1f}s")
        print(f"       Disqualified: {disqualified} (honeypots: {honeypots_found})")

    # ---------------------------------------------------------------
    # Step 4: Rank by score (descending), tiebreak by candidate_id (ascending)
    # ---------------------------------------------------------------
    if args.verbose:
        print("[4/5] Ranking candidates...")

    # Round scores to 4 decimal places BEFORE sorting so that
    # tie-breaking is consistent with the CSV output precision.
    for r in results:
        r['score'] = round(r['score'], 4)
    results.sort(key=lambda r: (-r['score'], r['candidate_id']))

    # Take top N
    top_n = results[:args.top_n]

    # Assign ranks
    for i, r in enumerate(top_n):
        r['rank'] = i + 1

    t_rank = time.time()

    # ---------------------------------------------------------------
    # Step 5: Generate reasoning and write CSV
    # ---------------------------------------------------------------
    if args.verbose:
        print(f"[5/5] Generating reasoning and writing {args.out}...")

    with open(args.out, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
        writer.writerow(['candidate_id', 'rank', 'score', 'reasoning'])

        for r in top_n:
            reasoning = generate_reasoning(r, r['rank'])
            writer.writerow([
                r['candidate_id'],
                r['rank'],
                f"{r['score']:.4f}",
                reasoning,
            ])

    t_end = time.time()

    # ---------------------------------------------------------------
    # Summary
    # ---------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"  Intelligent Candidate Ranking — Complete")
    print(f"{'='*60}")
    print(f"  Candidates loaded:      {len(candidates)}")
    print(f"  Disqualified:           {disqualified} ({honeypots_found} honeypots)")
    print(f"  Qualified & scored:     {len(results)}")
    print(f"  Top-{args.top_n} written to:    {args.out}")
    print(f"  Total time:             {t_end - t_start:.1f}s")
    print(f"{'='*60}")

    if args.verbose and top_n:
        print(f"\n  Top 5 candidates:")
        for r in top_n[:5]:
            d = r['details']
            ss = r['sub_scores']
            print(f"    #{r['rank']:>3} {r['candidate_id']} "
                  f"(Score: {r['score']:.4f}) "
                  f"[Sem:{ss['semantic']:.2f} Rule:{ss['rule_based']:.2f} "
                  f"Beh:{ss['behavioral']:.2f}]")
            print(f"         {d['current_title']} @ {d['current_company']}, "
                  f"{d['years_exp']:.1f}y, {d['location']}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
