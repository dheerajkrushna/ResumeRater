"""
scorer.py — Hybrid scoring engine combining rule-based, semantic, and behavioral scores.

Architecture:
  final_score = WEIGHT_SEMANTIC * semantic_score
              + WEIGHT_RULE_BASED * rule_score
              + WEIGHT_BEHAVIORAL * behavioral_score
  
  Then multiplied by location_multiplier and stability_multiplier.

The rule-based score itself is composed of:
  - Title alignment (15 pts)
  - Experience fit (15 pts)
  - Company & career trajectory (20 pts)
  - Technical skills + career keywords (35 pts)
"""

from src.config import (
    WEIGHT_SEMANTIC, WEIGHT_RULE_BASED, WEIGHT_BEHAVIORAL,
    BUDGET_TITLE, BUDGET_EXPERIENCE, BUDGET_COMPANY, BUDGET_SKILLS, BUDGET_TOTAL,
    TIER1_TITLES, TIER2_TITLES, TIER3_TITLES, TIER4_TITLES, ALL_ALLOWED_TITLES,
    TIER1_PRODUCT_COMPANIES, TIER2_PRODUCT_COMPANIES, SERVICES_COMPANIES,
    SKILL_WEIGHTS, PROFICIENCY_MULT,
    SKILLS_CORE_RETRIEVAL, SKILLS_CORE_RANKING, SKILLS_GENERAL_AI,
    SKILLS_CV_SPEECH, ALL_CAREER_KEYWORDS,
)
from src.honeypot_detector import detect_honeypot
from src.behavioral_scorer import score_behavioral, score_location


def _score_title(candidate):
    """Score based on current title alignment with JD. Returns (score, tier)."""
    title = (candidate.get('profile', {}).get('current_title') or '').lower()

    if title in TIER1_TITLES:
        return BUDGET_TITLE, 'tier1'
    elif title in TIER2_TITLES:
        return BUDGET_TITLE * 0.80, 'tier2'
    elif title in TIER3_TITLES:
        return BUDGET_TITLE * 0.55, 'tier3'
    elif title in TIER4_TITLES:
        return BUDGET_TITLE * 0.30, 'tier4'
    else:
        return 0.0, 'excluded'


def _score_experience(candidate):
    """Score based on years of experience vs JD ideal range (5-9, sweet spot 6-8)."""
    years = candidate.get('profile', {}).get('years_of_experience', 0)

    if 6.0 <= years <= 8.0:
        return BUDGET_EXPERIENCE  # Perfect
    elif 5.0 <= years <= 9.0:
        return BUDGET_EXPERIENCE * 0.85
    elif 4.0 <= years <= 11.0:
        return BUDGET_EXPERIENCE * 0.55
    elif 3.0 <= years <= 15.0:
        return BUDGET_EXPERIENCE * 0.25
    else:
        return 0.0


def _score_company_and_trajectory(candidate):
    """
    Score based on company types, career trajectory, and job stability.
    Returns (score, stability_multiplier, is_services_only).
    """
    career = candidate.get('career_history', [])
    profile = candidate.get('profile', {})

    if not career:
        return 0.0, 1.0, False

    # Classify each job
    services_count = 0
    product_t1_count = 0
    product_t2_count = 0
    total_jobs = len(career)

    for job in career:
        company = (job.get('company') or '').lower()
        if company in SERVICES_COMPANIES:
            services_count += 1
        elif company in TIER1_PRODUCT_COMPANIES:
            product_t1_count += 1
        elif company in TIER2_PRODUCT_COMPANIES:
            product_t2_count += 1

    # Services-only career → disqualify
    if services_count == total_jobs and total_jobs > 0:
        return 0.0, 1.0, True

    # Base score
    score = BUDGET_COMPANY * 0.35  # Baseline for having any career

    # Current company bonus
    current_company = (profile.get('current_company') or '').lower()
    if current_company in TIER1_PRODUCT_COMPANIES:
        score += BUDGET_COMPANY * 0.35
    elif current_company in TIER2_PRODUCT_COMPANIES:
        score += BUDGET_COMPANY * 0.25
    elif current_company in SERVICES_COMPANIES:
        score += BUDGET_COMPANY * 0.05

    # Product company diversity bonus
    product_total = product_t1_count + product_t2_count
    if product_total >= 3:
        score += BUDGET_COMPANY * 0.20
    elif product_total >= 2:
        score += BUDGET_COMPANY * 0.10

    # Tier 1 product experience extra bonus
    if product_t1_count >= 1:
        score += BUDGET_COMPANY * 0.05

    score = min(score, BUDGET_COMPANY)

    # ---------------------------------------------------------------
    # Career trajectory: title progression
    # ---------------------------------------------------------------
    titles = [job.get('title', '').lower() for job in career]
    seniority_keywords = ['senior', 'lead', 'staff', 'principal', 'head', 'director', 'vp']
    junior_keywords = ['junior', 'intern', 'trainee', 'fresher', 'associate']

    progression_signal = 0
    for i, t in enumerate(titles):
        if any(kw in t for kw in seniority_keywords):
            progression_signal += 1
        elif any(kw in t for kw in junior_keywords):
            progression_signal -= 1

    # Title-chaser detection: many different companies with short tenures
    # JD: "switching companies every 1.5 years — not a fit"
    durations = [job.get('duration_months', 0) for job in career]
    avg_duration = sum(durations) / len(durations) if durations else 0

    stability_mult = 1.0
    if total_jobs >= 3 and avg_duration < 18:
        stability_mult = 0.70  # Title-chaser penalty
    elif total_jobs >= 4 and avg_duration < 24:
        stability_mult = 0.85
    elif avg_duration >= 36:
        stability_mult = 1.10  # Stability bonus

    return score, stability_mult, False


def _score_skills_and_keywords(candidate):
    """
    Score technical skills using weighted categories and career description keywords.
    Returns (score, matched_skills_list, has_core_ir, cv_speech_only).
    """
    skills = candidate.get('skills', [])
    career = candidate.get('career_history', [])
    profile = candidate.get('profile', {})

    skill_points = 0.0
    matched_skills = []
    has_core_ir = False
    has_core_ranking = False
    has_general_ai = False
    has_cv_speech = False

    for s in skills:
        name = (s.get('name') or '').lower()
        proficiency = (s.get('proficiency') or 'beginner').lower()
        duration = s.get('duration_months', 0)

        weight = SKILL_WEIGHTS.get(name, 0.0)
        if weight <= 0:
            continue

        # Proficiency multiplier
        prof_m = PROFICIENCY_MULT.get(proficiency, 0.3)

        # Duration multiplier: longer experience → more credible
        dur_m = min((duration + 12) / 24.0, 2.0)

        # Endorsement credibility: skills with 0 endorsements and 0 assessment
        # scores are less trustworthy
        endorsements = s.get('endorsements', 0)
        endorse_m = 1.0
        if endorsements == 0 and proficiency in ('expert', 'advanced'):
            endorse_m = 0.6  # Unendorsed expert claim — discount

        points = weight * prof_m * dur_m * endorse_m
        skill_points += points
        matched_skills.append((name, points))

        # Track categories
        if name in SKILLS_CORE_RETRIEVAL:
            has_core_ir = True
        elif name in SKILLS_CORE_RANKING:
            has_core_ranking = True
        elif name in SKILLS_GENERAL_AI:
            has_general_ai = True
        elif name in SKILLS_CV_SPEECH:
            has_cv_speech = True

    # ---------------------------------------------------------------
    # Career description keyword bonus
    # ---------------------------------------------------------------
    desc_text = ' '.join(
        job.get('description', '') for job in career
    ).lower()
    summary = (profile.get('summary') or '').lower()
    full_text = desc_text + ' ' + summary

    keyword_bonus = 0.0
    for kw, pts in ALL_CAREER_KEYWORDS.items():
        if kw in full_text:
            keyword_bonus += pts

    skill_points += keyword_bonus

    # ---------------------------------------------------------------
    # CV/Speech-only penalty
    # JD: "primary expertise is computer vision, speech, or robotics
    #  without significant NLP/IR exposure"
    # ---------------------------------------------------------------
    cv_speech_only = False
    if has_cv_speech and not (has_core_ir or has_core_ranking or has_general_ai):
        cv_speech_only = True

    # Normalize to budget
    score = min(skill_points, BUDGET_SKILLS)

    # Sort matched skills by contribution
    matched_skills.sort(key=lambda x: -x[1])

    return score, matched_skills, (has_core_ir or has_core_ranking), cv_speech_only


def score_candidate(candidate, semantic_scores=None):
    """
    Compute the final hybrid score for a single candidate.

    Args:
        candidate: dict — candidate record
        semantic_scores: dict[str, float] — pre-computed TF-IDF scores (optional)

    Returns:
        dict with keys:
          - candidate_id, score, rank (to be filled later),
          - sub-scores, details, matched_skills, honeypot_reasons
          - disqualified (bool), disqualify_reason (str)
    """
    cand_id = candidate.get('candidate_id', '')
    profile = candidate.get('profile', {})
    result = {
        'candidate_id': cand_id,
        'candidate': candidate,
        'score': 0.0,
        'rank': 0,
        'disqualified': False,
        'disqualify_reason': '',
        'sub_scores': {},
        'details': {},
    }

    # ---------------------------------------------------------------
    # Step 0: Honeypot check
    # ---------------------------------------------------------------
    is_honeypot, honeypot_reasons = detect_honeypot(candidate)
    if is_honeypot:
        result['disqualified'] = True
        result['disqualify_reason'] = f"Honeypot: {'; '.join(honeypot_reasons)}"
        return result

    # ---------------------------------------------------------------
    # Step 1: Title filter
    # ---------------------------------------------------------------
    title = (profile.get('current_title') or '').lower()
    if title not in ALL_ALLOWED_TITLES:
        result['disqualified'] = True
        result['disqualify_reason'] = f"Non-tech title: '{profile.get('current_title')}'"
        return result

    # ---------------------------------------------------------------
    # Step 2: Rule-based scoring
    # ---------------------------------------------------------------
    title_score, title_tier = _score_title(candidate)
    exp_score = _score_experience(candidate)
    company_score, stability_mult, services_only = _score_company_and_trajectory(candidate)

    if services_only:
        result['disqualified'] = True
        result['disqualify_reason'] = "Entire career at consulting/services firms only"
        return result

    skills_score, matched_skills, has_core_ir, cv_speech_only = _score_skills_and_keywords(candidate)

    # CV/Speech penalty
    cv_mult = 0.30 if cv_speech_only else 1.0

    # Combine rule-based raw score
    raw_rule = title_score + exp_score + company_score + skills_score
    rule_score = min(raw_rule / BUDGET_TOTAL, 1.0)  # Normalize to [0, 1]

    # ---------------------------------------------------------------
    # Step 3: Semantic score (TF-IDF)
    # ---------------------------------------------------------------
    semantic_score = 0.0
    if semantic_scores and cand_id in semantic_scores:
        semantic_score = semantic_scores[cand_id]

    # ---------------------------------------------------------------
    # Step 4: Behavioral score
    # ---------------------------------------------------------------
    beh_score, beh_details = score_behavioral(candidate)
    loc_mult = score_location(candidate)

    # ---------------------------------------------------------------
    # Step 5: Combine into final score
    # ---------------------------------------------------------------
    weighted_score = (
        WEIGHT_SEMANTIC * semantic_score
        + WEIGHT_RULE_BASED * rule_score
        + WEIGHT_BEHAVIORAL * beh_score
    )

    # Apply multipliers
    final_score = weighted_score * stability_mult * cv_mult * loc_mult

    # Clamp to [0, 1]
    final_score = max(0.0, min(final_score, 1.0))

    # ---------------------------------------------------------------
    # Store results
    # ---------------------------------------------------------------
    result['score'] = final_score
    result['sub_scores'] = {
        'semantic': semantic_score,
        'rule_based': rule_score,
        'behavioral': beh_score,
        'title': title_score / BUDGET_TITLE,
        'experience': exp_score / BUDGET_EXPERIENCE,
        'company': company_score / BUDGET_COMPANY,
        'skills': skills_score / BUDGET_SKILLS,
    }
    result['details'] = {
        'title_tier': title_tier,
        'stability_mult': stability_mult,
        'cv_mult': cv_mult,
        'location_mult': loc_mult,
        'behavioral': beh_details,
        'matched_skills': matched_skills[:10],
        'has_core_ir': has_core_ir,
        'cv_speech_only': cv_speech_only,
        'years_exp': profile.get('years_of_experience', 0),
        'current_title': profile.get('current_title', ''),
        'current_company': profile.get('current_company', ''),
        'location': profile.get('location', ''),
        'country': profile.get('country', ''),
    }

    return result
