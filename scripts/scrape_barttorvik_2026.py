"""
Scrape 2025-26 game results from Barttorvik using requests with proper session handling.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import requests
import time
from src import config
from src.logger import setup_logger

logger = setup_logger(__name__)


def fetch_barttorvik_games_csv(year: int = 2026, retry_delay: int = 5, max_retries: int = 3):
    """
    Fetch game data from Barttorvik CSV export.

    Args:
        year: Season year (2026 for 2025-26 season)
        retry_delay: Seconds to wait between retries
        max_retries: Maximum number of retry attempts
    """
    # CSV export URL
    url = f"https://barttorvik.com/gamestat.php?year={year}&csv=1"

    logger.info(f"Fetching games from Barttorvik for {year} season...")
    logger.info(f"URL: {url}")

    # Create a session with realistic browser headers
    session = requests.Session()
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
        'Accept-Encoding': 'gzip, deflate, br',
        'DNT': '1',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1',
        'Sec-Fetch-Dest': 'document',
        'Sec-Fetch-Mode': 'navigate',
        'Sec-Fetch-Site': 'none',
        'Cache-Control': 'max-age=0',
    })

    for attempt in range(max_retries):
        try:
            logger.info(f"Attempt {attempt + 1}/{max_retries}...")

            # Make request with timeout
            response = session.get(url, timeout=30)
            response.raise_for_status()

            # Check if we got actual CSV data (not a loading page)
            content = response.text

            if 'year,month,day' in content or 'team,opponent' in content:
                logger.info("✓ Successfully fetched CSV data")

                # Parse CSV
                from io import StringIO
                df = pd.read_csv(StringIO(content))

                logger.info(f"✓ Loaded {len(df)} rows")
                logger.info(f"  Columns: {df.columns.tolist()}")

                return df

            elif 'Verifying' in content or 'browser' in content.lower():
                logger.warning(f"Got verification page, retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
                continue

            else:
                logger.warning(f"Unexpected response content (first 200 chars):")
                logger.warning(content[:200])
                time.sleep(retry_delay)
                continue

        except requests.RequestException as e:
            logger.error(f"Request failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
            continue

    logger.error(f"Failed to fetch data after {max_retries} attempts")
    return None


def main():
    logger.info("="*70)
    logger.info("SCRAPING BARTTORVIK 2025-26 GAMES")
    logger.info("="*70)

    # Try to fetch data
    df = fetch_barttorvik_games_csv(year=2026, retry_delay=5, max_retries=3)

    if df is None or df.empty:
        logger.error("\n❌ Could not fetch data from Barttorvik")
        logger.info("\nAlternative options:")
        logger.info("1. Manually download from browser:")
        logger.info("   - Open: https://barttorvik.com/gamestat.php?year=2026")
        logger.info("   - Wait for page to load")
        logger.info("   - Click 'Export to CSV' or similar button")
        logger.info("   - Save as: data/raw/games/2025-26_results_raw.csv")
        logger.info("\n2. Try ESPN or Sports-Reference instead")
        logger.info("\n3. Skip 2025-26 validation for now (2024-25 validation is sufficient)")
        return

    # Save raw data
    output_path = config.DATA_DIR / "raw" / "games" / "2025-26_results_raw.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    logger.info(f"\n✓ Saved raw data to: {output_path}")
    logger.info(f"\nNext steps:")
    logger.info("  python scripts/convert_barttorvik_2026_games.py")


if __name__ == "__main__":
    main()
