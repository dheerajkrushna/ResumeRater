# 🎯 Intelligent Candidate Discovery & Ranking System

**Redrob Hackathon — India Runs Data & AI Challenge**

An AI-powered candidate ranking system that goes beyond keyword matching to understand who genuinely fits the **Senior AI Engineer — Founding Team** role. Uses a hybrid scoring approach combining semantic similarity, rule-based feature extraction, and behavioral signal analysis.

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   candidates.jsonl                       │
└──────────────────────┬──────────────────────────────────┘
                       │
              ┌────────▼────────┐
              │  Honeypot Filter │ ──► Discard impossible profiles
              └────────┬────────┘
                       │
         ┌─────────────┼─────────────┐
         │             │             │
   ┌─────▼─────┐ ┌────▼────┐ ┌─────▼──────┐
   │  TF-IDF   │ │  Rule   │ │ Behavioral │
   │  Semantic  │ │  Based  │ │  Signals   │
   │  (30%)    │ │  (50%)  │ │  (20%)     │
   └─────┬─────┘ └────┬────┘ └─────┬──────┘
         │             │             │
         └─────────────┼─────────────┘
                       │
              ┌────────▼────────┐
              │  × Location Fit  │
              │  × Stability     │
              │  × CV-only Pen.  │
              └────────┬────────┘
                       │
              ┌────────▼────────┐
              │  Rank + Reason  │
              └────────┬────────┘
                       │
              ┌────────▼────────┐
              │  submission.csv  │
              └─────────────────┘
```

### Scoring Components

| Component | Weight | What it measures |
|-----------|--------|------------------|
| **TF-IDF Semantic** | 30% | Cosine similarity between JD text and candidate profile documents using TF-IDF vectorization |
| **Title Alignment** | ~8.8% | How closely the current title matches the Senior AI Engineer role (4 tiers) |
| **Experience Fit** | ~8.8% | Years of experience vs. the JD's ideal range of 5-9 years (sweet spot: 6-8) |
| **Company & Trajectory** | ~11.8% | Product vs. services career, company tier, stability, title progression |
| **Technical Skills** | ~20.6% | Weighted skill scoring across 7 categories + career description keyword bonus |
| **Behavioral Signals** | 20% | Activity recency, response rate, notice period, GitHub, platform engagement |

### Multipliers

| Multiplier | Range | Purpose |
|------------|-------|---------|
| **Location** | 0.05–1.0 | Prefer India / preferred cities / relocation willingness |
| **Stability** | 0.70–1.10 | Penalize title-chasers (<18mo avg), reward stability (>36mo avg) |
| **CV-only** | 0.30 | Penalize candidates with only CV/Speech skills and no NLP/IR |

---

## 🛡️ Honeypot Detection

The system detects synthetically impossible profiles through 5 strategies:

1. **Skill Duration Anomaly**: Expert/advanced proficiency in 3+ skills with 0 months of use
2. **Experience Timeline**: Years of experience exceeds total career span
3. **Education/Career Overlap**: Started working 6+ years before starting education
4. **Job Duration Discrepancy**: Stated duration exceeds the start→end date span
5. **Keyword Stuffer**: Non-tech title listing many AI skills with no AI work in descriptions

---

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- pip

### Setup

```bash
git clone https://github.com/YOUR_USERNAME/redrob-ranker.git
cd redrob-ranker
pip install -r requirements.txt
```

### Reproduce Submission

```bash
python rank.py --candidates ./candidates.jsonl --out ./submission.csv --verbose
```

This runs the full pipeline on 100,000 candidates and produces `submission.csv` with the top 100 ranked candidates. Typical runtime: **60-90 seconds** on a standard CPU.

### Run the Streamlit Demo

```bash
pip install streamlit
streamlit run app.py
```

### Run Evaluation

```bash
python evaluation/evaluate.py --submission ./submission.csv --candidates ./candidates.jsonl
```

---

## 📁 Project Structure

```
.
├── rank.py                        # CLI entry point
├── app.py                         # Streamlit sandbox demo
├── requirements.txt               # Python dependencies
├── submission.csv                  # Generated submission
├── submission_metadata.yaml       # Team metadata
├── README.md                      # This file
│
├── src/
│   ├── __init__.py
│   ├── config.py                  # All constants, weights, skill categories
│   ├── loader.py                  # JSONL/GZ data loading + text extraction
│   ├── honeypot_detector.py       # Impossible profile detection (5 strategies)
│   ├── semantic_scorer.py         # TF-IDF vectorizer + cosine similarity
│   ├── behavioral_scorer.py       # Platform engagement signal scoring
│   ├── scorer.py                  # Hybrid scoring engine (combines all)
│   └── reasoning_generator.py     # Dynamic fact-checked reasoning text
│
└── evaluation/
    └── evaluate.py                # NDCG, MAP, P@K computation + analysis
```

---

## 🧠 Design Decisions

### Why TF-IDF over Transformer Embeddings?
The compute constraint (CPU-only, 5 minutes, 16GB RAM) rules out transformer-based embeddings for 100K candidates. TF-IDF with `scikit-learn` processes the entire corpus in under 10 seconds and captures domain-specific n-grams ("vector search", "hybrid retrieval") effectively.

### Why Rule-Based + Semantic (Hybrid)?
Pure semantic matching can't capture structural signals the JD explicitly mentions: company type (product vs. services), experience range, career stability, notice period. Pure rule-based matching misses candidates who describe relevant work differently. The hybrid approach combines both strengths.

### Why Weight Title Higher Than Skills?
The JD explicitly warns: *"A candidate who has all the AI keywords listed as skills but whose title is 'Marketing Manager' is not a fit."* Title is a more trustworthy signal of what someone actually does professionally than a self-reported skill list.

### Why Behavioral Signals as a Separate Component?
A perfect-on-paper candidate who hasn't logged in for 6 months and has a 5% response rate is not actually hireable. Behavioral signals (response rate, notice period, activity recency) measure *availability*, which is orthogonal to *fit*.

### What Would Change with GPU + API Access?
With more compute, we would:
1. Use `sentence-transformers/all-MiniLM-L6-v2` for dense embeddings
2. Implement a two-stage retrieval pipeline (BM25 recall → reranker precision)
3. Use an LLM (e.g., Gemma-2B) for reasoning generation instead of templates
4. Add cross-encoder reranking for the top 500 candidates

---

## ⚙️ Compute Profile

| Metric | Value |
|--------|-------|
| **Runtime** | ~60-90s on Intel i7, ~40s on M2 |
| **Peak RAM** | ~400MB |
| **GPU** | Not required |
| **Network** | Not required |
| **Disk** | ~500MB (candidates.jsonl) |

---

## 📊 Evaluation Methodology

We validate the ranking through:

1. **Format Validation**: `validate_submission.py` passes with 0 errors
2. **Honeypot Check**: 0 honeypots in top 100 (detection rate: ~171/100K flagged)
3. **Title Sanity**: All top 100 candidates have technical titles
4. **Reasoning Quality**: Manual review of 10 sampled reasonings for:
   - Specific facts from profile ✓
   - JD requirement connection ✓
   - Honest concern acknowledgment ✓
   - No hallucination ✓
   - Variation between candidates ✓
5. **Score Monotonicity**: Verified non-increasing scores with rank

---

## 🤖 AI Tools Declaration

- **Gemini**: Used for architecture discussion, code review, and iterative development
- All candidate data processing and ranking runs locally with no AI API calls

---

## 📜 License

MIT License — built for the Redrob Hackathon 2026.
