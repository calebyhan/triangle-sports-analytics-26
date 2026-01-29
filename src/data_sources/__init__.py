"""
Data source modules for NCAA basketball analytics.

This package provides unified interfaces to various college basketball data sources:
- ESPN: BPI ratings, team stats, player stats
- Haslametrics: Momentum metrics, alternative ratings
- CBBpy: Play-by-play, box scores, game metadata

All modules follow consistent patterns:
- Rate limiting built-in
- Automatic caching
- Standardized output formats
- Error handling and retries
"""

from . import espn
from . import haslametrics
from . import cbbpy_enhanced

__all__ = ['espn', 'haslametrics', 'cbbpy_enhanced']
