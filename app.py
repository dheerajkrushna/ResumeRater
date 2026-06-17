"""
app.py — Streamlit sandbox demo for the Intelligent Candidate Ranking System.

This provides a web interface where organizers can:
1. Upload a small candidate sample (JSON/JSONL)
2. Run the ranking pipeline end-to-end
3. View ranked results with scores and reasoning
4. Inspect individual candidate profiles

Deploy to Streamlit Cloud (free tier) for the sandbox requirement.
"""

import streamlit as st
import json
import csv
import io
import sys
import os
import time

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.loader import extract_candidate_text
from src.semantic_scorer import SemanticScorer
from src.scorer import score_candidate
from src.reasoning_generator import generate_reasoning
from src.honeypot_detector import detect_honeypot


st.set_page_config(
    page_title="Redrob AI — Candidate Ranker",
    page_icon="🎯",
    layout="wide",
)


def run_ranking(candidates):
    """Run the full ranking pipeline on a list of candidates."""
    progress = st.progress(0, text="Initializing...")

    # Step 1: Semantic scoring
    progress.progress(20, text="Computing TF-IDF semantic scores...")
    semantic_scorer = SemanticScorer()
    semantic_scores = semantic_scorer.fit_and_score(candidates)

    # Step 2: Score all candidates
    progress.progress(50, text="Scoring candidates...")
    results = []
    disqualified = []

    for cand in candidates:
        result = score_candidate(cand, semantic_scores=semantic_scores)
        if result['disqualified']:
            disqualified.append(result)
        elif result['score'] > 0:
            results.append(result)

    # Step 3: Rank
    progress.progress(80, text="Ranking and generating reasoning...")
    results.sort(key=lambda r: (-r['score'], r['candidate_id']))

    # Assign ranks and generate reasoning
    for i, r in enumerate(results[:100]):
        r['rank'] = i + 1
        r['reasoning'] = generate_reasoning(r, r['rank'])

    progress.progress(100, text="Done!")
    return results[:100], disqualified


def main():
    st.title("🎯 Intelligent Candidate Discovery & Ranking")
    st.markdown("**Redrob AI — Senior AI Engineer Role Matching System**")
    st.markdown("---")

    # Sidebar
    with st.sidebar:
        st.header("📋 About")
        st.markdown("""
        This system ranks candidates for the **Senior AI Engineer — Founding Team** 
        role using a hybrid scoring approach:
        
        - 🔍 **TF-IDF Semantic Matching** (30%)
        - 📊 **Rule-Based Feature Scoring** (50%)
        - 📈 **Behavioral Signal Analysis** (20%)
        
        Multiplied by location fit and stability multipliers.
        """)

        st.header("⚠️ Honeypot Detection")
        st.markdown("""
        Automatically detects impossible profiles:
        - Skill duration anomalies
        - Experience timeline contradictions
        - Education/career overlaps
        - Keyword stuffer patterns
        """)

    # File upload
    st.header("1️⃣ Upload Candidates")
    uploaded_file = st.file_uploader(
        "Upload a candidate file (JSON array or JSONL, ≤100 candidates)",
        type=['json', 'jsonl'],
    )

    # Also provide a way to use the sample file
    use_sample = st.checkbox("Or use the bundled sample_candidates.json (50 candidates)")

    candidates = None

    if uploaded_file:
        content = uploaded_file.read().decode('utf-8')
        try:
            # Try JSON array first
            candidates = json.loads(content)
            if not isinstance(candidates, list):
                candidates = [candidates]
        except json.JSONDecodeError:
            # Try JSONL
            candidates = []
            for line in content.strip().split('\n'):
                if line.strip():
                    candidates.append(json.loads(line))
        st.success(f"Loaded {len(candidates)} candidates")

    elif use_sample:
        sample_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "[PUB] India_runs_data_and_ai_challenge",
            "India_runs_data_and_ai_challenge",
            "sample_candidates.json"
        )
        if os.path.exists(sample_path):
            with open(sample_path, 'r', encoding='utf-8') as f:
                candidates = json.load(f)
            st.success(f"Loaded {len(candidates)} sample candidates")
        else:
            st.error(f"Sample file not found at {sample_path}")

    if candidates:
        # Run ranking
        st.header("2️⃣ Ranking Results")

        if st.button("🚀 Run Ranking Pipeline", type="primary"):
            t_start = time.time()
            ranked, disqualified = run_ranking(candidates)
            t_elapsed = time.time() - t_start

            st.metric("Processing Time", f"{t_elapsed:.1f}s")

            col1, col2, col3 = st.columns(3)
            col1.metric("Candidates Processed", len(candidates))
            col2.metric("Qualified & Ranked", len(ranked))
            col3.metric("Disqualified", len(disqualified))

            if ranked:
                st.subheader("🏆 Ranked Candidates")

                # Display as table
                table_data = []
                for r in ranked:
                    d = r['details']
                    ss = r['sub_scores']
                    table_data.append({
                        'Rank': r['rank'],
                        'ID': r['candidate_id'],
                        'Score': f"{r['score']:.4f}",
                        'Title': d.get('current_title', ''),
                        'Company': d.get('current_company', ''),
                        'Exp (yrs)': f"{d.get('years_exp', 0):.1f}",
                        'Location': d.get('location', ''),
                        'Semantic': f"{ss.get('semantic', 0):.2f}",
                        'Rule': f"{ss.get('rule_based', 0):.2f}",
                        'Behavioral': f"{ss.get('behavioral', 0):.2f}",
                        'Reasoning': r.get('reasoning', ''),
                    })

                st.dataframe(table_data, use_container_width=True, height=500)

                # Download CSV
                csv_buf = io.StringIO()
                writer = csv.writer(csv_buf)
                writer.writerow(['candidate_id', 'rank', 'score', 'reasoning'])
                for r in ranked:
                    writer.writerow([
                        r['candidate_id'],
                        r['rank'],
                        f"{r['score']:.4f}",
                        r.get('reasoning', ''),
                    ])

                st.download_button(
                    label="📥 Download Ranking CSV",
                    data=csv_buf.getvalue(),
                    file_name="ranked_candidates.csv",
                    mime="text/csv",
                )

            if disqualified:
                with st.expander(f"🚫 Disqualified Candidates ({len(disqualified)})"):
                    for d in disqualified:
                        st.markdown(
                            f"- **{d['candidate_id']}**: {d['disqualify_reason']}"
                        )

            # Store in session state for detailed inspection
            st.session_state['ranked'] = ranked

        # Detailed candidate inspector
        if 'ranked' in st.session_state and st.session_state['ranked']:
            st.header("3️⃣ Candidate Inspector")
            ranked = st.session_state['ranked']
            selected_id = st.selectbox(
                "Select a candidate to inspect",
                [f"#{r['rank']} — {r['candidate_id']} ({r['details']['current_title']})"
                 for r in ranked]
            )

            if selected_id:
                idx = int(selected_id.split('#')[1].split(' ')[0]) - 1
                r = ranked[idx]
                candidate = r['candidate']
                profile = candidate['profile']
                signals = candidate.get('redrob_signals', {})

                col1, col2 = st.columns(2)

                with col1:
                    st.subheader("Profile")
                    st.markdown(f"**{profile.get('anonymized_name')}**")
                    st.markdown(f"*{profile.get('headline')}*")
                    st.markdown(f"📍 {profile.get('location')}, {profile.get('country')}")
                    st.markdown(f"🏢 {profile.get('current_company')} ({profile.get('current_industry')})")
                    st.markdown(f"📅 {profile.get('years_of_experience', 0):.1f} years experience")
                    st.markdown(f"\n{profile.get('summary', '')}")

                    st.subheader("Career History")
                    for job in candidate.get('career_history', []):
                        current = "🟢 Current" if job.get('is_current') else ""
                        st.markdown(
                            f"**{job.get('title')}** at {job.get('company')} "
                            f"({job.get('duration_months', 0)} months) {current}"
                        )
                        st.markdown(f"_{job.get('description', '')[:200]}..._")

                with col2:
                    st.subheader("Scoring Breakdown")
                    ss = r['sub_scores']
                    for k, v in ss.items():
                        st.progress(v, text=f"{k}: {v:.2f}")

                    st.subheader("Behavioral Signals")
                    st.json({
                        'Response Rate': f"{signals.get('recruiter_response_rate', 0):.0%}",
                        'Notice Period': f"{signals.get('notice_period_days', 0)} days",
                        'Open to Work': signals.get('open_to_work_flag', False),
                        'GitHub Score': signals.get('github_activity_score', -1),
                        'Last Active': signals.get('last_active_date', 'N/A'),
                    })

                    st.subheader("Top Matched Skills")
                    for name, pts in r['details'].get('matched_skills', [])[:8]:
                        st.markdown(f"- **{name}** ({pts:.1f} pts)")

                st.subheader("Reasoning")
                st.info(r.get('reasoning', 'N/A'))


if __name__ == '__main__':
    main()
