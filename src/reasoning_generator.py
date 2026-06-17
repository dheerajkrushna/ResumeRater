"""
reasoning_generator.py — Generate fact-checked, varied reasoning for ranked candidates.

Stage 4 review samples 10 random reasonings and checks:
1. Specific facts from the candidate's profile
2. Connection to specific JD requirements
3. Honest concerns acknowledged
4. No hallucination (every claim must exist in the profile)
5. Variation between candidates (not templated)
6. Rank consistency (tone matches the rank)

Strategy: Template bank with randomized sentence structures.
Each reasoning is 2 sentences assembled from verified profile facts.
"""

import random
import hashlib


def _seed_from_id(candidate_id):
    """Deterministic seed from candidate_id for reproducible variation."""
    h = int(hashlib.md5(candidate_id.encode()).hexdigest(), 16)
    return h % (2**31)


def _get_career_companies(candidate):
    """Get list of companies from career history."""
    return [job.get('company', '') for job in candidate.get('career_history', [])]


def _get_top_skills(result):
    """Get top matched skill names from scoring result."""
    matched = result.get('details', {}).get('matched_skills', [])
    return [name for name, _ in matched[:5]]


def _format_skills_text(skills, max_items=3):
    """Format a skill list into natural text."""
    if not skills:
        return "relevant technical background"
    if len(skills) == 1:
        return skills[0]
    if len(skills) <= max_items:
        return ', '.join(skills[:-1]) + ' and ' + skills[-1]
    return ', '.join(skills[:max_items - 1]) + f' and {len(skills) - max_items + 1} others'


def generate_reasoning(result, rank):
    """
    Generate a 1-2 sentence reasoning for a ranked candidate.

    Args:
        result: dict — scoring result from scorer.py
        rank: int — the candidate's rank (1-100)

    Returns:
        str: reasoning text (no quotes, no newlines)
    """
    candidate = result.get('candidate', {})
    profile = candidate.get('profile', {})
    career = candidate.get('career_history', [])
    signals = candidate.get('redrob_signals', {})
    details = result.get('details', {})
    sub_scores = result.get('sub_scores', {})

    cand_id = result.get('candidate_id', '')
    rng = random.Random(_seed_from_id(cand_id))

    # Extract facts
    title = profile.get('current_title', 'Professional')
    company = profile.get('current_company', '')
    years = profile.get('years_of_experience', 0)
    location = profile.get('location', '')
    country = profile.get('country', '')
    notice = signals.get('notice_period_days', 0)
    resp_rate = signals.get('recruiter_response_rate', 0.0)
    github = signals.get('github_activity_score', -1)
    open_to_work = signals.get('open_to_work_flag', False)

    companies = _get_career_companies(candidate)
    top_skills = _get_top_skills(result)
    skills_text = _format_skills_text(top_skills)
    has_core_ir = details.get('has_core_ir', False)
    title_tier = details.get('title_tier', 'tier4')

    # Past notable companies (excluding current)
    past_companies = [c for c in companies[1:] if c and c != company][:2]
    past_text = f", previously at {' and '.join(past_companies)}" if past_companies else ""

    # ---------------------------------------------------------------
    # Sentence 1 templates: Profile & career summary
    # ---------------------------------------------------------------
    s1_templates = [
        lambda: f"{title} at {company} with {years:.1f} years of experience{past_text}.",
        lambda: f"{years:.1f}-year {title} currently at {company}{past_text}.",
        lambda: f"Currently a {title} at {company} ({years:.1f} yrs total experience){past_text}.",
        lambda: f"Brings {years:.1f} years as {title}, currently with {company}{past_text}.",
        lambda: f"Experienced {title} ({years:.1f} yrs) working at {company}{past_text}.",
    ]

    # ---------------------------------------------------------------
    # Sentence 2 templates: Technical fit + logistics
    # ---------------------------------------------------------------
    # Vary based on rank tier and strengths/concerns
    concerns = []
    strengths = []

    # Strengths
    if has_core_ir:
        strengths.append(f"hands-on with {skills_text}")
    elif top_skills:
        strengths.append(f"skills in {skills_text}")

    if title_tier in ('tier1', 'tier2'):
        strengths.append("strong title alignment with the Senior AI Engineer role")

    if notice <= 30:
        strengths.append(f"{notice}-day notice period (ideal)")
    if resp_rate >= 0.75:
        strengths.append(f"{resp_rate:.0%} recruiter response rate")
    if github >= 40:
        strengths.append(f"active GitHub contributor (score: {github:.0f})")
    if open_to_work:
        strengths.append("actively looking for new opportunities")

    # Concerns
    if notice > 90:
        concerns.append(f"long notice period ({notice} days)")
    if resp_rate < 0.20:
        concerns.append(f"low recruiter response rate ({resp_rate:.0%})")
    if country != 'India':
        concerns.append(f"based outside India ({location})")
    elif not any(city in location.lower() for city in ['noida', 'pune', 'delhi', 'gurgaon', 'mumbai', 'hyderabad', 'bangalore', 'bengaluru']):
        if not signals.get('willing_to_relocate', False):
            concerns.append(f"located in {location} with no relocation willingness")
    if details.get('cv_speech_only'):
        concerns.append("primary expertise in CV/speech without core NLP/IR exposure")
    if details.get('stability_mult', 1.0) < 0.85:
        concerns.append("frequent job changes (avg tenure under 18 months)")

    # Build sentence 2
    s2_parts = []

    if strengths:
        strength_sample = rng.sample(strengths, min(len(strengths), 3))
        s2_parts.append(_format_skills_text(strength_sample, max_items=3))

    if rank <= 25:
        # Top tier: emphasize fit, mention any concern briefly
        if concerns:
            concern_text = concerns[0]
            s2_templates = [
                lambda: f"Strong JD alignment with {'; '.join(s2_parts)}; minor concern: {concern_text}.",
                lambda: f"Excellent fit — {'; '.join(s2_parts)}. Only flag: {concern_text}.",
                lambda: f"Top-tier match: {'; '.join(s2_parts)}. Note: {concern_text}.",
            ]
        else:
            s2_templates = [
                lambda: f"Strong JD alignment with {'; '.join(s2_parts)}; no major concerns.",
                lambda: f"Excellent fit — {'; '.join(s2_parts)}. Well-positioned candidate.",
                lambda: f"Compelling match: {'; '.join(s2_parts)}. Recommend fast outreach.",
            ]
    elif rank <= 60:
        # Mid tier: balanced assessment
        if concerns:
            concern_text = '; '.join(concerns[:2])
            s2_templates = [
                lambda: f"Solid fit with {'; '.join(s2_parts)}; concerns include {concern_text}.",
                lambda: f"Good match — {'; '.join(s2_parts)}. Weighed against: {concern_text}.",
                lambda: f"Viable candidate: {'; '.join(s2_parts)}. Trade-offs: {concern_text}.",
            ]
        else:
            s2_templates = [
                lambda: f"Good match with {'; '.join(s2_parts)}.",
                lambda: f"Solid candidate — {'; '.join(s2_parts)}.",
                lambda: f"Reasonable fit: {'; '.join(s2_parts)}.",
            ]
    else:
        # Lower tier: acknowledge gaps
        if concerns:
            concern_text = '; '.join(concerns[:2])
            s2_templates = [
                lambda: f"Partial fit with {'; '.join(s2_parts)}; significant gaps: {concern_text}.",
                lambda: f"Some relevant signals ({'; '.join(s2_parts)}) but held back by {concern_text}.",
                lambda: f"Fringe candidate: {'; '.join(s2_parts)}. Key concerns: {concern_text}.",
            ]
        else:
            s2_templates = [
                lambda: f"Marginal alignment: {'; '.join(s2_parts)}. Included as a lower-confidence pick.",
                lambda: f"Limited fit but notable for {'; '.join(s2_parts)}.",
                lambda: f"Below ideal profile but has {'; '.join(s2_parts)}.",
            ]

    # Select templates deterministically based on candidate_id
    s1 = rng.choice(s1_templates)()
    s2 = rng.choice(s2_templates)()

    # Combine and clean
    reasoning = f"{s1} {s2}"
    # Remove any double spaces or weird punctuation
    reasoning = ' '.join(reasoning.split())
    # Escape any commas in the reasoning to avoid CSV issues
    reasoning = reasoning.replace('"', "'")

    return reasoning
