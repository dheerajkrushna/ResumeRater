"""
behavioral_scorer.py — Score candidates based on platform engagement signals.

A perfect-on-paper candidate who hasn't logged in for 6 months and has a
5% recruiter response rate is, for hiring purposes, not actually available.

Scoring dimensions:
1. Activity recency (last_active_date)
2. Open-to-work flag
3. Recruiter response rate & speed
4. Notice period
5. GitHub activity
6. Platform engagement (saved by recruiters, profile views)
7. Profile verification signals
8. Location / relocation fit
"""

from datetime import datetime
from src.config import (
    REFERENCE_DATE,
    PREFERRED_LOCATIONS_INDIA, OTHER_INDIA_CITIES,
    RECENCY_EXCELLENT_DAYS, RECENCY_GOOD_DAYS,
    RESPONSE_RATE_EXCELLENT, RESPONSE_RATE_GOOD, RESPONSE_RATE_POOR,
    RESPONSE_TIME_FAST_HOURS, RESPONSE_TIME_OK_HOURS,
    NOTICE_PERIOD_IDEAL, NOTICE_PERIOD_OK, NOTICE_PERIOD_LONG,
    GITHUB_ACTIVE_THRESHOLD,
)


def _parse_date(date_str):
    if not date_str:
        return None
    try:
        return datetime.strptime(date_str, "%Y-%m-%d")
    except (ValueError, TypeError):
        return None


def score_behavioral(candidate):
    """
    Compute a behavioral engagement score for a candidate.

    Args:
        candidate: dict — a single candidate record

    Returns:
        tuple: (score: float in [0, 1], details: dict with sub-scores)
    """
    signals = candidate.get('redrob_signals', {})
    profile = candidate.get('profile', {})

    points = 0.0
    max_points = 0.0
    details = {}

    # ---------------------------------------------------------------
    # 1. Activity Recency (max 5 points)
    # ---------------------------------------------------------------
    max_points += 5.0
    last_active = _parse_date(signals.get('last_active_date'))
    if last_active:
        days_inactive = (REFERENCE_DATE - last_active).days
        if days_inactive <= RECENCY_EXCELLENT_DAYS:
            points += 5.0
            details['recency'] = f"Active {days_inactive}d ago (excellent)"
        elif days_inactive <= RECENCY_GOOD_DAYS:
            points += 3.0
            details['recency'] = f"Active {days_inactive}d ago (good)"
        elif days_inactive <= 365:
            points += 1.0
            details['recency'] = f"Active {days_inactive}d ago (stale)"
        else:
            details['recency'] = f"Inactive for {days_inactive}d (dormant)"
    else:
        details['recency'] = "No activity data"

    # ---------------------------------------------------------------
    # 2. Open to Work (max 2 points)
    # ---------------------------------------------------------------
    max_points += 2.0
    if signals.get('open_to_work_flag'):
        points += 2.0
        details['open_to_work'] = True
    else:
        details['open_to_work'] = False

    # ---------------------------------------------------------------
    # 3. Recruiter Response Rate (max 4 points, can go negative)
    # ---------------------------------------------------------------
    max_points += 4.0
    resp_rate = signals.get('recruiter_response_rate', 0.0)
    if resp_rate >= RESPONSE_RATE_EXCELLENT:
        points += 4.0
        details['response_rate'] = f"{resp_rate:.0%} (excellent)"
    elif resp_rate >= RESPONSE_RATE_GOOD:
        points += 2.0
        details['response_rate'] = f"{resp_rate:.0%} (good)"
    elif resp_rate >= RESPONSE_RATE_POOR:
        points += 0.5
        details['response_rate'] = f"{resp_rate:.0%} (low)"
    else:
        points -= 2.0  # Penalty for near-zero response
        details['response_rate'] = f"{resp_rate:.0%} (very low — availability concern)"

    # ---------------------------------------------------------------
    # 4. Response Time (max 2 points)
    # ---------------------------------------------------------------
    max_points += 2.0
    resp_time = signals.get('avg_response_time_hours', 999)
    if resp_time <= RESPONSE_TIME_FAST_HOURS:
        points += 2.0
        details['response_time'] = f"{resp_time:.0f}h (fast)"
    elif resp_time <= RESPONSE_TIME_OK_HOURS:
        points += 1.0
        details['response_time'] = f"{resp_time:.0f}h (reasonable)"
    else:
        details['response_time'] = f"{resp_time:.0f}h (slow)"

    # ---------------------------------------------------------------
    # 5. Notice Period (max 4 points, can go negative)
    # ---------------------------------------------------------------
    max_points += 4.0
    notice = signals.get('notice_period_days', 180)
    if notice <= NOTICE_PERIOD_IDEAL:
        points += 4.0
        details['notice_period'] = f"{notice}d (ideal — under 30d)"
    elif notice <= NOTICE_PERIOD_OK:
        points += 2.0
        details['notice_period'] = f"{notice}d (acceptable)"
    elif notice <= NOTICE_PERIOD_LONG:
        points += 0.0
        details['notice_period'] = f"{notice}d (long but manageable)"
    else:
        points -= 2.0
        details['notice_period'] = f"{notice}d (very long — hiring friction)"

    # ---------------------------------------------------------------
    # 6. GitHub Activity (max 2 points)
    # ---------------------------------------------------------------
    max_points += 2.0
    github = signals.get('github_activity_score', -1)
    if github >= GITHUB_ACTIVE_THRESHOLD:
        points += 2.0
        details['github'] = f"Score {github} (active contributor)"
    elif github >= 10:
        points += 1.0
        details['github'] = f"Score {github} (some activity)"
    elif github == -1:
        details['github'] = "No GitHub linked"
    else:
        details['github'] = f"Score {github} (minimal activity)"

    # ---------------------------------------------------------------
    # 7. Platform Engagement (max 2 points)
    # ---------------------------------------------------------------
    max_points += 2.0
    saved = signals.get('saved_by_recruiters_30d', 0)
    views = signals.get('profile_views_received_30d', 0)
    interview_rate = signals.get('interview_completion_rate', 0.0)

    engagement_pts = 0.0
    if saved >= 5:
        engagement_pts += 0.5
    if views >= 20:
        engagement_pts += 0.5
    if interview_rate >= 0.8:
        engagement_pts += 1.0
    elif interview_rate >= 0.5:
        engagement_pts += 0.5
    points += min(engagement_pts, 2.0)
    details['engagement'] = (
        f"Saved:{saved}, Views:{views}, "
        f"Interview rate:{interview_rate:.0%}"
    )

    # ---------------------------------------------------------------
    # 8. Verification Signals (max 1 point)
    # ---------------------------------------------------------------
    max_points += 1.0
    verified = 0
    if signals.get('verified_email'):
        verified += 1
    if signals.get('verified_phone'):
        verified += 1
    if signals.get('linkedin_connected'):
        verified += 1
    points += min(verified / 3.0, 1.0)
    details['verified'] = f"{verified}/3 verified"

    # ---------------------------------------------------------------
    # Normalize to [0, 1]
    # ---------------------------------------------------------------
    score = max(0.0, min(points / max_points, 1.0))
    details['raw_points'] = points
    details['max_points'] = max_points

    return score, details


def score_location(candidate):
    """
    Compute a location fit multiplier.

    Returns:
        float: multiplier in [0.1, 1.0]
    """
    profile = candidate.get('profile', {})
    signals = candidate.get('redrob_signals', {})

    location = (profile.get('location') or '').lower()
    country = (profile.get('country') or '').lower()
    willing = signals.get('willing_to_relocate', False)

    if country != 'india':
        return 0.15 if willing else 0.05

    # Check if in preferred Indian cities
    is_preferred = any(city in location for city in PREFERRED_LOCATIONS_INDIA)
    if is_preferred:
        return 1.0

    # Other Indian cities
    is_other_india = any(city in location for city in OTHER_INDIA_CITIES)
    if is_other_india:
        return 0.85 if willing else 0.65

    # Unknown Indian location
    return 0.75 if willing else 0.55
