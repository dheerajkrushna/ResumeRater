"""
honeypot_detector.py — Detect synthetically impossible candidate profiles.

The dataset contains ~80 honeypot candidates with subtly impossible profiles.
Including them in the top 100 is penalized (>10% = disqualification).

Detection strategies:
1. Skill duration anomaly: expert/advanced proficiency with 0 months of use
2. Experience timeline anomaly: years_of_experience exceeds career span
3. Education/career overlap: started working years before starting education
4. Job duration discrepancy: duration_months exceeds start→end span
5. Keyword stuffer detection: AI skills listed but no AI work in descriptions
"""

from datetime import datetime
from src.config import (
    REFERENCE_DATE, SKILLS_CORE_RETRIEVAL, SKILLS_CORE_RANKING,
    SKILLS_GENERAL_AI, SKILLS_NON_TECH
)


def _parse_date(date_str):
    """Parse a YYYY-MM-DD date string, returning None on failure."""
    if not date_str:
        return None
    try:
        return datetime.strptime(date_str, "%Y-%m-%d")
    except (ValueError, TypeError):
        return None


def detect_honeypot(candidate):
    """
    Check if a candidate profile is a honeypot (synthetically impossible).

    Args:
        candidate: dict — a single candidate record

    Returns:
        tuple: (is_honeypot: bool, reasons: list[str])
    """
    reasons = []
    profile = candidate.get('profile', {})
    career = candidate.get('career_history', [])
    skills = candidate.get('skills', [])
    education = candidate.get('education', [])

    # ---------------------------------------------------------------
    # Check 1: Expert/advanced skills with 0 duration_months
    # Honeypot signal: "expert" in 10 skills with 0 years used
    # ---------------------------------------------------------------
    high_prof_skills = [
        s for s in skills
        if s.get('proficiency') in ('expert', 'advanced')
    ]
    zero_dur_experts = [
        s for s in high_prof_skills
        if s.get('duration_months', 0) == 0
    ]
    if len(zero_dur_experts) >= 3:
        skill_names = [s['name'] for s in zero_dur_experts]
        reasons.append(
            f"Claims {len(zero_dur_experts)} expert/advanced skills with 0 months usage: "
            f"{', '.join(skill_names[:5])}"
        )

    # ---------------------------------------------------------------
    # Check 2: Years of experience exceeds career timeline
    # e.g., "13.7 years experience" but first job started 1.3 years ago
    # ---------------------------------------------------------------
    years_exp = profile.get('years_of_experience', 0)
    start_dates = []
    for job in career:
        sd = _parse_date(job.get('start_date'))
        if sd:
            start_dates.append(sd)

    if start_dates:
        first_job = min(start_dates)
        max_possible = (REFERENCE_DATE - first_job).days / 365.25
        if years_exp > max_possible + 0.5:
            reasons.append(
                f"Claims {years_exp:.1f} years experience but career spans only "
                f"{max_possible:.1f} years (first job: {first_job.strftime('%Y-%m-%d')})"
            )

    # ---------------------------------------------------------------
    # Check 3: First job predates education by >6 years
    # e.g., started working in 2011 but started college in 2019
    # ---------------------------------------------------------------
    if education and start_dates:
        edu_start_years = [
            e.get('start_year') for e in education
            if e.get('start_year')
        ]
        if edu_start_years:
            min_edu_year = min(edu_start_years)
            min_job_year = min(sd.year for sd in start_dates)
            if min_edu_year - min_job_year > 6:
                reasons.append(
                    f"Started working in {min_job_year} but started education in "
                    f"{min_edu_year} (gap of {min_edu_year - min_job_year} years)"
                )

    # ---------------------------------------------------------------
    # Check 4: Job duration_months exceeds start→end date span
    # ---------------------------------------------------------------
    for i, job in enumerate(career):
        start = _parse_date(job.get('start_date'))
        end = _parse_date(job.get('end_date'))
        dur = job.get('duration_months', 0)
        if start and end:
            span = (end.year - start.year) * 12 + (end.month - start.month)
            if dur > span + 6:
                reasons.append(
                    f"Job at {job.get('company')}: claims {dur} months but "
                    f"start→end span is only {span} months"
                )

    # ---------------------------------------------------------------
    # Check 5: Title-skill incoherence (keyword stuffer detection)
    # A non-tech professional listing many AI skills is suspicious,
    # especially when career descriptions don't mention any AI work.
    # ---------------------------------------------------------------
    current_title = (profile.get('current_title') or '').lower()
    non_tech_titles = {
        'marketing manager', 'hr manager', 'accountant', 'sales executive',
        'content writer', 'graphic designer', 'operations manager',
        'civil engineer', 'mechanical engineer', 'customer support',
        'project manager',
    }

    if current_title in non_tech_titles:
        ai_skills = set()
        for s in skills:
            sname = s.get('name', '').lower()
            if sname in SKILLS_CORE_RETRIEVAL | SKILLS_CORE_RANKING | SKILLS_GENERAL_AI:
                ai_skills.add(sname)

        if len(ai_skills) >= 5:
            # Check if descriptions actually mention any AI work
            desc_text = ' '.join(
                job.get('description', '') for job in career
            ).lower()
            ai_desc_keywords = [
                'model', 'embedding', 'vector', 'neural', 'transformer',
                'nlp', 'search', 'ranking', 'recommendation', 'ml ',
                'machine learning', 'deep learning', 'retrieval',
            ]
            ai_desc_matches = sum(1 for kw in ai_desc_keywords if kw in desc_text)
            if ai_desc_matches < 2:
                reasons.append(
                    f"Keyword stuffer: '{profile.get('current_title')}' lists "
                    f"{len(ai_skills)} AI skills but career descriptions contain "
                    f"no AI/ML work"
                )

    is_honeypot = len(reasons) > 0
    return is_honeypot, reasons
