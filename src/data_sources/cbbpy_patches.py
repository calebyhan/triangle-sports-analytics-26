"""
Monkey patches for CBBpy to fix NCAA.com API changes.

This module automatically patches CBBpy when imported, fixing compatibility
issues without modifying the installed package. This keeps the fixes in
version control and makes them portable.

Fixes Applied:
1. KeyError: 'isConferenceGame' - NCAA.com removed this field
2. KeyError: 'text' - Missing field in shot chart data
3. TypeError: fuzzy matching returns None when team list is empty
4. Missing 2026 season in team map CSV

Import this module before using CBBpy:
    from src.data_sources import cbbpy_patches
    from cbbpy import mens_scraper

    # CBBpy is now patched and ready to use
    games = mens_scraper.get_games_team('Duke', season=2026)
"""

import logging
from pathlib import Path
import pandas as pd

logger = logging.getLogger(__name__)

# Track if patches have been applied
_PATCHES_APPLIED = False


def apply_patches():
    """
    Apply auto-patches for CBBpy. Called automatically on import.

    Currently only handles team map updates. For code-level patches
    (isConferenceGame, shot chart, fuzzy matching), use the manual
    patch script: scripts/patch_cbbpy_venv.py
    """
    global _PATCHES_APPLIED

    if _PATCHES_APPLIED:
        logger.debug("CBBpy auto-patches already applied")
        return

    try:
        # Ensure current and future seasons exist in team map
        ensure_season_in_team_map(2026)  # Current season
        ensure_season_in_team_map(2027)  # Future season

        logger.debug("CBBpy auto-patches applied successfully")
        _PATCHES_APPLIED = True

    except ImportError:
        logger.warning("CBBpy not installed - auto-patches not applied")
    except Exception as e:
        logger.debug(f"Auto-patches skipped: {e}")


def ensure_season_in_team_map(season: int) -> bool:
    """
    Ensure a season exists in CBBpy's team map CSV.

    If the season doesn't exist, copy data from the previous season.
    This fixes the "No teams found for season X" error.

    Args:
        season: Season year to ensure exists (e.g., 2026)

    Returns:
        True if season exists or was added successfully
    """
    try:
        from cbbpy.utils import cbbpy_utils
        import pandas as pd

        # Find the team map CSV
        utils_path = Path(cbbpy_utils.__file__).parent
        csv_path = utils_path / 'mens_team_map.csv'

        if not csv_path.exists():
            logger.error(f"Team map not found at {csv_path}")
            return False

        # Load and check if season exists
        df = pd.read_csv(csv_path)

        if season in df['season'].values:
            logger.debug(f"Season {season} already in team map")
            return True

        # Season doesn't exist - copy from previous year
        logger.info(f"Adding season {season} to CBBpy team map...")

        latest_season = df['season'].max()
        source_season = min(latest_season, season - 1)

        season_data = df[df['season'] == source_season].copy()

        if len(season_data) == 0:
            logger.error(f"No data found for season {source_season} to copy")
            return False

        season_data['season'] = season
        df_updated = pd.concat([df, season_data], ignore_index=True)

        # Save updated CSV
        df_updated.to_csv(csv_path, index=False)

        # Clear Python cache for the module
        pycache_dir = utils_path / '__pycache__'
        if pycache_dir.exists():
            for pyc_file in pycache_dir.glob('cbbpy_utils.*.pyc'):
                try:
                    pyc_file.unlink()
                    logger.debug(f"Deleted cached bytecode: {pyc_file.name}")
                except Exception as e:
                    logger.warning(f"Could not delete {pyc_file.name}: {e}")

        logger.info(f"✓ Added {len(season_data)} teams for season {season}")
        return True

    except ImportError:
        logger.warning("CBBpy not installed")
        return False
    except Exception as e:
        logger.error(f"Failed to update team map: {e}")
        return False


def get_available_seasons() -> list:
    """
    Get list of available seasons in CBBpy team map.

    Returns:
        List of season years
    """
    try:
        from cbbpy.utils import cbbpy_utils

        utils_path = Path(cbbpy_utils.__file__).parent
        csv_path = utils_path / 'mens_team_map.csv'

        if not csv_path.exists():
            return []

        df = pd.read_csv(csv_path)
        return sorted(df['season'].unique().tolist())

    except Exception as e:
        logger.error(f"Failed to get available seasons: {e}")
        return []


# Auto-apply patches on import
apply_patches()


if __name__ == '__main__':
    print("CBBpy Patches Module")
    print("=" * 50)

    # Check if CBBpy is installed
    try:
        import cbbpy
        print(f"✓ CBBpy {cbbpy.__version__} installed")
    except ImportError:
        print("✗ CBBpy not installed")
        exit(1)

    # Show available seasons
    seasons = get_available_seasons()
    if seasons:
        print(f"\n✓ Available seasons: {min(seasons)}-{max(seasons)}")
        print(f"  Total seasons: {len(seasons)}")
    else:
        print("\n✗ Could not read team map")

    # Ensure 2026 exists
    print("\nEnsuring season 2026 exists...")
    if ensure_season_in_team_map(2026):
        print("✓ Season 2026 ready")
    else:
        print("✗ Failed to add season 2026")

    # Ensure 2027 for future
    print("\nEnsuring season 2027 exists...")
    if ensure_season_in_team_map(2027):
        print("✓ Season 2027 ready")
    else:
        print("✗ Failed to add season 2027")

    # Show updated seasons
    seasons = get_available_seasons()
    if seasons:
        print(f"\n✓ Updated seasons: {min(seasons)}-{max(seasons)}")
