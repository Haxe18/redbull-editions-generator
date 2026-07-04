#!/usr/bin/env python3
"""Shared language priority lookup for Red Bull locale domains.

Single source of truth for the language preference order used by both the
collector (merging editions from multiple language versions per country) and
the processor (selecting the preferred raw file per country).
"""

# Priority order: English first, then German/Dutch, then other European languages.
_LANGUAGE_PRIORITIES = {
    "en": 1,  # English
    "gb": 1,  # UK English
    "us": 1,  # US English
    "de": 2,  # German
    "nl": 2,  # Dutch
    "at": 3,  # Austrian German
    "ch": 3,  # Swiss German
    "fr": 4,  # French
    "es": 5,  # Spanish
    "pt": 6,  # Portuguese
    "it": 7,  # Italian
}


def get_language_priority(domain: str) -> int:
    """Get language priority for a locale domain (lower = higher priority).

    Args:
        domain: The locale domain (e.g., 'us-en', 'ch-de').

    Returns:
        Priority number (lower = higher priority); 99 for unknown languages,
        999 for malformed domains.
    """
    if not domain or "-" not in domain:
        return 999

    lang = domain.split("-")[-1].lower()
    return _LANGUAGE_PRIORITIES.get(lang, 99)
