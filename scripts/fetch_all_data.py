"""
Fetch data from all configured sources.

This script fetches data from ESPN, Haslametrics, CBBpy, and Sports-Reference
based on configuration settings in config.py.

Usage:
    python scripts/fetch_all_data.py --year 2026 --sources all
    python scripts/fetch_all_data.py --year 2026 --sources espn,haslametrics
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import pandas as pd
from src import config
from src.data_sources import espn, haslametrics, cbbpy_enhanced
from src.logger import setup_logger

logger = setup_logger(__name__)


def fetch_espn_data(year: int) -> pd.DataFrame:
    """Fetch data from ESPN API"""
    if not config.ESPN_CONFIG['enabled']:
        logger.info("ESPN is disabled in config")
        return pd.DataFrame()

    logger.info(f"Fetching ESPN data for {year}...")

    try:
        # Fetch standings (contains basic team stats)
        standings = espn.fetch_standings(year=year)

        if len(standings) > 0:
            espn.save_team_data(standings, f'standings_{year}.csv')
            logger.info(f"✓ Saved {len(standings)} teams from ESPN")
            return standings
        else:
            logger.warning("No ESPN data retrieved")
            return pd.DataFrame()

    except Exception as e:
        logger.error(f"Failed to fetch ESPN data: {e}")
        return pd.DataFrame()


def fetch_haslametrics_data(year: int) -> tuple:
    """Fetch data from Haslametrics"""
    if not config.HASLAMETRICS_CONFIG['enabled']:
        logger.info("Haslametrics is disabled in config")
        return pd.DataFrame(), pd.DataFrame()

    logger.info(f"Fetching Haslametrics data for {year}...")

    try:
        # Import the Selenium scraper for proper DOM extraction
        from src.data_sources.haslametrics_selenium import fetch_haslametrics_table

        # Fetch ratings using DOM extraction (not pd.read_html)
        ratings = fetch_haslametrics_table(year)
        if len(ratings) > 0:
            # Save directly without cleaning (already clean from DOM extraction)
            output_path = config.HASLAMETRICS_DATA_DIR / f'ratings_{year}.csv'
            ratings.to_csv(output_path, index=False)
            logger.info(f"✓ Saved {len(ratings)} team ratings from Haslametrics")

        # Fetch momentum
        momentum = haslametrics.fetch_momentum_metrics(year)
        if len(momentum) > 0:
            haslametrics.save_haslametrics_data(momentum, f'momentum_{year}.csv')
            logger.info(f"✓ Saved {len(momentum)} momentum records from Haslametrics")

        return ratings, momentum

    except Exception as e:
        logger.error(f"Failed to fetch Haslametrics data: {e}")
        return pd.DataFrame(), pd.DataFrame()


def fetch_cbbpy_data(year: int, teams: list = None) -> pd.DataFrame:
    """Fetch data from CBBpy"""
    if not config.CBBPY_CONFIG['enabled']:
        logger.info("CBBpy is disabled in config")
        return pd.DataFrame()

    if not cbbpy_enhanced.CBBPY_AVAILABLE:
        logger.warning("CBBpy not installed - skipping")
        return pd.DataFrame()

    logger.info(f"Fetching CBBpy data for {year}...")

    try:
        # If specific teams provided, fetch those
        if teams:
            all_games = []
            for team in teams:
                logger.info(f"  Fetching games for {team}...")
                result = cbbpy_enhanced.fetch_games_team(team, season=year, include_all=False)
                if isinstance(result, pd.DataFrame) and len(result) > 0:
                    all_games.append(result)

            if all_games:
                combined = pd.concat(all_games, ignore_index=True)
                cbbpy_enhanced.save_cbbpy_data(combined, f'games_{year}.csv')
                logger.info(f"✓ Saved {len(combined)} games from CBBpy")
                return combined

        else:
            # Fetch all games for the season (may be slow)
            logger.warning("Fetching ALL games - this may take a while...")
            games = cbbpy_enhanced.fetch_games_season(season=year)
            if len(games) > 0:
                cbbpy_enhanced.save_cbbpy_data(games, f'all_games_{year}.csv')
                logger.info(f"✓ Saved {len(games)} games from CBBpy")
                return games

        return pd.DataFrame()

    except Exception as e:
        logger.error(f"Failed to fetch CBBpy data: {e}")
        return pd.DataFrame()


def main():
    parser = argparse.ArgumentParser(description='Fetch NCAA basketball data from all sources')
    parser.add_argument('--year', type=int, default=config.PREDICTION_YEAR,
                        help='Season year (default: from config)')
    parser.add_argument('--sources', type=str, default='all',
                        help='Comma-separated list of sources: espn,haslametrics,cbbpy (default: all)')
    parser.add_argument('--teams', type=str, default=None,
                        help='Comma-separated list of teams for CBBpy (default: all)')

    args = parser.parse_args()

    # Parse sources
    if args.sources == 'all':
        sources = ['espn', 'haslametrics', 'cbbpy']
    else:
        sources = [s.strip() for s in args.sources.split(',')]

    # Parse teams
    teams = None
    if args.teams:
        teams = [t.strip() for t in args.teams.split(',')]

    logger.info("="*60)
    logger.info("NCAA BASKETBALL DATA COLLECTION")
    logger.info("="*60)
    logger.info(f"Year: {args.year}")
    logger.info(f"Sources: {', '.join(sources)}")
    if teams:
        logger.info(f"Teams: {', '.join(teams)}")
    logger.info("="*60)

    # Fetch data from each source
    results = {}

    if 'espn' in sources:
        logger.info("\n[1/4] ESPN API")
        results['espn'] = fetch_espn_data(args.year)

    if 'haslametrics' in sources:
        logger.info("\n[2/4] Haslametrics")
        ratings, momentum = fetch_haslametrics_data(args.year)
        results['haslametrics_ratings'] = ratings
        results['haslametrics_momentum'] = momentum

    if 'cbbpy' in sources:
        logger.info("\n[3/4] CBBpy")
        results['cbbpy'] = fetch_cbbpy_data(args.year, teams=teams)

    # Summary
    logger.info("\n" + "="*60)
    logger.info("COLLECTION SUMMARY")
    logger.info("="*60)
    for source, data in results.items():
        if isinstance(data, pd.DataFrame):
            status = "✓" if len(data) > 0 else "✗"
            count = len(data) if len(data) > 0 else 0
            logger.info(f"{status} {source:25s} {count:5d} records")
    logger.info("="*60)

    logger.info("\n✓ Data collection complete!")
    logger.info(f"   Check data/raw/ directories for output files")


if __name__ == '__main__':
    main()
