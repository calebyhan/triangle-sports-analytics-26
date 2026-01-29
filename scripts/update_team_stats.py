"""
Update team statistics from all configured data sources.

This script updates the main team_stats file used for predictions by merging
data from Barttorvik (core) with optional sources (ESPN, Haslametrics, etc.)

Usage:
    python scripts/update_team_stats.py --year 2026
    python scripts/update_team_stats.py --year 2026 --enable-bpi
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import pandas as pd
from src import config
from src.utils import fetch_barttorvik_year
from src.data_sources import espn, haslametrics
from src.logger import setup_logger

logger = setup_logger(__name__)


def fetch_core_stats(year: int) -> pd.DataFrame:
    """Fetch core team statistics from Barttorvik"""
    logger.info(f"Fetching core stats from Barttorvik for {year}...")

    try:
        df = fetch_barttorvik_year(year)

        # Standardize column names
        df = df.rename(columns={
            'team': 'team',
            'adjoe': 'adj_oe',
            'adjde': 'adj_de',
            'barthag': 'barthag',
        })

        # Calculate derived metrics
        df['adj_em'] = df['adj_oe'] - df['adj_de']
        df['off_efficiency'] = df['adj_oe']
        df['def_efficiency'] = df['adj_de']
        df['power_rating'] = df['adj_em']

        # Approximate other metrics
        df['ppg'] = df['adj_oe'] * 0.70
        df['opp_ppg'] = df['adj_de'] * 0.70
        df['pace'] = 70.0  # Default pace

        # Calculate win percentage if record is available
        if 'record' in df.columns:
            # Parse "W-L" format
            df[['wins', 'losses']] = df['record'].str.split('-', expand=True).astype(float)
            df['win_pct'] = df['wins'] / (df['wins'] + df['losses'])
        else:
            df['win_pct'] = 0.5

        logger.info(f"✓ Fetched {len(df)} teams from Barttorvik")
        return df

    except Exception as e:
        logger.error(f"Failed to fetch Barttorvik data: {e}")
        raise


def add_espn_data(df: pd.DataFrame, year: int) -> pd.DataFrame:
    """Add ESPN BPI ratings if available"""
    if not config.ESPN_CONFIG['enabled']:
        return df

    logger.info("Adding ESPN BPI ratings...")

    try:
        # Try to load cached ESPN data
        espn_file = config.ESPN_DATA_DIR / f'standings_{year}.csv'
        if espn_file.exists():
            espn_df = pd.read_csv(espn_file)

            # Try to extract BPI-like metric if available
            # This is placeholder - actual BPI extraction needs implementation
            logger.warning("ESPN BPI extraction not yet implemented")

            return df
        else:
            logger.warning("No cached ESPN data found - run fetch_all_data.py first")
            return df

    except Exception as e:
        logger.error(f"Failed to add ESPN data: {e}")
        return df


def add_haslametrics_data(df: pd.DataFrame, year: int) -> pd.DataFrame:
    """Add Haslametrics momentum metrics if available"""
    if not config.HASLAMETRICS_CONFIG['enabled']:
        return df

    logger.info("Adding Haslametrics momentum metrics...")

    try:
        # Try to load cached Haslametrics data
        momentum_file = config.HASLAMETRICS_DATA_DIR / f'momentum_{year}.csv'
        if momentum_file.exists():
            momentum_df = pd.read_csv(momentum_file)

            # Standardize team names
            if 'team' in momentum_df.columns:
                momentum_df['team'] = momentum_df['team'].apply(haslametrics.standardize_team_name)

                # Merge with main data
                df = df.merge(
                    momentum_df,
                    on='team',
                    how='left',
                    suffixes=('', '_hasla')
                )

                logger.info(f"✓ Added momentum metrics for {len(df)} teams")
        else:
            logger.warning("No cached Haslametrics data found - run fetch_all_data.py first")

        return df

    except Exception as e:
        logger.error(f"Failed to add Haslametrics data: {e}")
        return df


def filter_to_target_teams(df: pd.DataFrame, teams: list = None) -> pd.DataFrame:
    """Filter to specific teams if needed"""
    if teams is None:
        # Use default ACC + relevant teams
        teams = [
            'Duke', 'North Carolina', 'NC State', 'Virginia', 'Virginia Tech',
            'Miami', 'Florida State', 'Clemson', 'Georgia Tech', 'Boston College',
            'Syracuse', 'Louisville', 'Pittsburgh', 'Notre Dame', 'Wake Forest',
            'California', 'Stanford', 'SMU', 'Ohio State', 'Michigan', 'Baylor'
        ]

    # Standardize team names in DataFrame
    df['team_clean'] = df['team'].apply(lambda x: x.replace('Florida St.', 'Florida State')
                                                    .replace('Miami FL', 'Miami')
                                                    .replace('N.C. State', 'NC State')
                                                    .replace('Ohio St.', 'Ohio State')
                                                    .replace('Pittsburgh', 'Pittsburgh'))

    # Filter to target teams
    filtered = df[df['team_clean'].isin(teams)].copy()

    # Use cleaned name
    filtered['team'] = filtered['team_clean']
    filtered = filtered.drop(columns=['team_clean'], errors='ignore')

    logger.info(f"Filtered to {len(filtered)} target teams")
    missing = set(teams) - set(filtered['team'].tolist())
    if missing:
        logger.warning(f"Missing teams: {missing}")

    return filtered


def main():
    parser = argparse.ArgumentParser(description='Update team statistics from all sources')
    parser.add_argument('--year', type=int, default=config.PREDICTION_YEAR,
                        help='Season year (default: from config)')
    parser.add_argument('--enable-bpi', action='store_true',
                        help='Include ESPN BPI data')
    parser.add_argument('--enable-momentum', action='store_true',
                        help='Include Haslametrics momentum data')
    parser.add_argument('--teams', type=str, default=None,
                        help='Comma-separated list of teams to include (default: all)')

    args = parser.parse_args()

    logger.info("="*60)
    logger.info("TEAM STATISTICS UPDATE")
    logger.info("="*60)
    logger.info(f"Year: {args.year}")
    logger.info(f"ESPN BPI: {'enabled' if args.enable_bpi else 'disabled'}")
    logger.info(f"Haslametrics Momentum: {'enabled' if args.enable_momentum else 'disabled'}")
    logger.info("="*60)

    # Fetch core statistics
    logger.info("\n[1/3] Fetching core statistics from Barttorvik...")
    team_stats = fetch_core_stats(args.year)

    # Add optional data sources
    if args.enable_bpi:
        logger.info("\n[2/3] Adding ESPN BPI data...")
        team_stats = add_espn_data(team_stats, args.year)
    else:
        logger.info("\n[2/3] Skipping ESPN BPI (disabled)")

    if args.enable_momentum:
        logger.info("\n[3/3] Adding Haslametrics momentum data...")
        team_stats = add_haslametrics_data(team_stats, args.year)
    else:
        logger.info("\n[3/3] Skipping Haslametrics momentum (disabled)")

    # Filter to target teams if specified
    if args.teams:
        teams = [t.strip() for t in args.teams.split(',')]
        team_stats = filter_to_target_teams(team_stats, teams)
    else:
        team_stats = filter_to_target_teams(team_stats)

    # Select final columns
    base_columns = [
        'team', 'off_efficiency', 'def_efficiency', 'ppg', 'opp_ppg',
        'pace', 'power_rating', 'win_pct'
    ]

    # Add optional columns if they exist
    optional_columns = [col for col in team_stats.columns if col not in base_columns]
    final_columns = base_columns + optional_columns

    team_stats = team_stats[final_columns]

    # Save to processed directory
    output_file = config.PROCESSED_DATA_DIR / f'team_stats_{args.year-1}_{args.year%100:02d}.csv'
    team_stats.to_csv(output_file, index=False)

    logger.info("\n" + "="*60)
    logger.info("UPDATE SUMMARY")
    logger.info("="*60)
    logger.info(f"Teams: {len(team_stats)}")
    logger.info(f"Columns: {len(final_columns)}")
    logger.info(f"Output: {output_file}")
    logger.info("="*60)

    logger.info("\n✓ Team statistics updated!")
    logger.info(f"\nNext steps:")
    logger.info(f"  1. Review data: head {output_file}")
    logger.info(f"  2. Run modeling: jupyter notebook notebooks/02_modeling.ipynb")


if __name__ == '__main__':
    main()
