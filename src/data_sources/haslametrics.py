"""
Haslametrics scraper for college basketball analytics.

Haslametrics provides free alternative efficiency ratings with unique metrics:
- Team ratings and rankings
- Momentum scores (recent performance trends)
- Offensive rebound scoring frequency
- Steal scoring frequency
- Consistency metrics

Website: https://www.haslametrics.com
All data is free and scrapeable (be respectful with rate limiting)

NOTE: Haslametrics uses JavaScript to render data dynamically, so we use
Selenium with headless Chrome to scrape the fully-rendered page.
"""

import requests
import pandas as pd
import time
from pathlib import Path
from typing import Optional, Dict
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
from ..config import HASLAMETRICS_CONFIG, HASLAMETRICS_DATA_DIR, CACHE_DIR
from ..logger import setup_logger

logger = setup_logger(__name__)

# Try to import Selenium (optional dependency)
try:
    from selenium import webdriver
    from selenium.webdriver.chrome.service import Service
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    from webdriver_manager.chrome import ChromeDriverManager
    SELENIUM_AVAILABLE = True
except ImportError:
    SELENIUM_AVAILABLE = False
    logger.warning("Selenium not available - install with: pip install selenium webdriver-manager")

# Ensure directories exist
HASLAMETRICS_DATA_DIR.mkdir(parents=True, exist_ok=True)

BASE_URL = HASLAMETRICS_CONFIG['base_url']
TIMEOUT = HASLAMETRICS_CONFIG['timeout']
CACHE_TTL = HASLAMETRICS_CONFIG['cache_ttl']


class RateLimiter:
    """Rate limiter for Haslametrics requests"""
    def __init__(self, requests_per_minute: int = 30):
        self.requests_per_minute = requests_per_minute
        self.requests = []

    def wait_if_needed(self):
        """Wait if we've exceeded rate limit"""
        now = datetime.now()
        cutoff = now - timedelta(minutes=1)

        self.requests = [r for r in self.requests if r > cutoff]

        if len(self.requests) >= self.requests_per_minute:
            sleep_time = (self.requests[0] - cutoff).total_seconds() + 0.1
            logger.info(f"Rate limit reached, sleeping {sleep_time:.1f}s")
            time.sleep(sleep_time)
            self.requests = []

        self.requests.append(now)


_rate_limiter = RateLimiter(HASLAMETRICS_CONFIG['rate_limit'])


def _fetch_page(url: str, use_cache: bool = True) -> Optional[str]:
    """
    Fetch HTML page with rate limiting and caching.

    Args:
        url: URL to fetch
        use_cache: Whether to use caching

    Returns:
        HTML content as string
    """
    # Check cache
    cache_key = url.replace('/', '_').replace(':', '_').replace('?', '_')
    cache_path = CACHE_DIR / 'haslametrics' / f"{cache_key}.html"

    if use_cache and cache_path.exists():
        age = time.time() - cache_path.stat().st_mtime
        if age < CACHE_TTL:
            logger.debug(f"Loading from cache: {cache_key}")
            return cache_path.read_text()

    # Fetch from website
    _rate_limiter.wait_if_needed()

    try:
        logger.debug(f"Fetching: {url}")
        headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
        }
        response = requests.get(url, headers=headers, timeout=TIMEOUT)
        response.raise_for_status()

        html = response.text

        # Save to cache
        if use_cache:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(html)

        return html

    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to fetch {url}: {e}")
        return None


def _get_webdriver():
    """Get a headless Chrome webdriver"""
    if not SELENIUM_AVAILABLE:
        raise ImportError("Selenium not installed. Install with: pip install selenium webdriver-manager")

    chrome_options = Options()
    chrome_options.add_argument("--headless")  # Run in background
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--window-size=1920,1080")

    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=chrome_options)
    return driver


def fetch_team_ratings_selenium(year: int = 2026) -> pd.DataFrame:
    """
    Fetch team ratings from Haslametrics using Selenium (JavaScript-rendered page).

    Args:
        year: Season year (e.g., 2026 for 2025-26 season)

    Returns:
        DataFrame with team ratings
    """
    if not SELENIUM_AVAILABLE:
        logger.error("Selenium not available - cannot fetch Haslametrics data")
        return pd.DataFrame()

    url = f"{BASE_URL}/ratings.php?yr={year}"

    logger.info(f"Fetching Haslametrics ratings with Selenium for {year}...")

    driver = None
    try:
        driver = _get_webdriver()
        driver.get(url)

        # Wait for the table to be populated by JavaScript
        wait = WebDriverWait(driver, 20)
        wait.until(EC.presence_of_element_located((By.ID, "myTable")))

        # Give extra time for JavaScript to populate all cells
        time.sleep(3)

        # Get the fully-rendered HTML
        html = driver.page_source

        # Use custom parser instead of pd.read_html
        from .haslametrics_parser import parse_haslametrics_ratings
        df = parse_haslametrics_ratings(html)

        if df.empty:
            logger.warning(f"No data parsed from {url}")
            return pd.DataFrame()

        logger.info(f"Fetched {len(df)} team ratings from Haslametrics via Selenium")
        return df

    except Exception as e:
        logger.error(f"Failed to fetch Haslametrics ratings with Selenium: {e}")
        return pd.DataFrame()

    finally:
        if driver:
            driver.quit()


def fetch_team_ratings(year: int = 2026, use_selenium: bool = True) -> pd.DataFrame:
    """
    Fetch team ratings from Haslametrics.

    Args:
        year: Season year (e.g., 2026 for 2025-26 season)
        use_selenium: Use Selenium for JavaScript rendering (recommended)

    Returns:
        DataFrame with team ratings
    """
    # Haslametrics requires Selenium since data is loaded via JavaScript
    if use_selenium and SELENIUM_AVAILABLE:
        return fetch_team_ratings_selenium(year)

    # Fallback to simple HTTP (will return empty table cells)
    logger.warning("Using fallback HTTP fetch - may return empty data due to JavaScript")

    url = f"{BASE_URL}/ratings.php?yr={year}"
    html = _fetch_page(url)

    if html is None:
        logger.error(f"Failed to fetch ratings for {year}")
        return pd.DataFrame()

    try:
        # Parse HTML with pandas read_html
        tables = pd.read_html(html)

        if len(tables) == 0:
            logger.warning(f"No tables found in {url}")
            return pd.DataFrame()

        # Haslametrics has multiple tables - find the largest one (main ratings table)
        df = max(tables, key=len)

        logger.info(f"Fetched {len(df)} team ratings from Haslametrics")
        return df

    except Exception as e:
        logger.error(f"Failed to parse Haslametrics ratings: {e}")
        return pd.DataFrame()


def fetch_momentum_metrics(year: int = 2026) -> pd.DataFrame:
    """
    Fetch momentum metrics from Haslametrics.

    Momentum metrics show recent performance trends and can help
    identify teams that are improving or declining.

    Args:
        year: Season year

    Returns:
        DataFrame with momentum scores
    """
    # Try different possible URLs for momentum data
    possible_urls = [
        f"{BASE_URL}/momentum.php?yr={year}",
        f"{BASE_URL}/ratings.php?yr={year}",  # Might be on main ratings page
    ]

    for url in possible_urls:
        html = _fetch_page(url)

        if html is None:
            continue

        try:
            tables = pd.read_html(html)

            # Look for a table with "momentum" or "trend" columns
            for table in tables:
                cols_lower = [str(col).lower() for col in table.columns]
                if any('momentum' in col or 'trend' in col for col in cols_lower):
                    logger.info(f"Fetched {len(table)} momentum records from {url}")
                    return table

        except Exception as e:
            logger.debug(f"Failed to parse {url}: {e}")
            continue

    logger.warning(f"Could not find momentum data for {year} - may not be available on Haslametrics")
    return pd.DataFrame()


def fetch_consistency_metrics(year: int = 2026) -> pd.DataFrame:
    """
    Fetch consistency metrics from Haslametrics.

    Consistency metrics show how reliable a team's performance is,
    which is useful for identifying teams that blow out weak opponents
    vs. teams that play close games.

    Args:
        year: Season year

    Returns:
        DataFrame with consistency metrics
    """
    # This is a placeholder - actual endpoint may vary
    # Haslametrics may have this under a different page
    logger.warning("Consistency metrics endpoint not yet implemented")
    return pd.DataFrame(columns=['team', 'consistency_score', 'variance'])


def standardize_team_name(haslametrics_name: str) -> str:
    """
    Convert Haslametrics team name to standard format.

    Args:
        haslametrics_name: Team name from Haslametrics

    Returns:
        Standardized team name
    """
    # Haslametrics typically uses standard names, but may have variations
    mappings = {
        'North Carolina': 'North Carolina',
        'NC State': 'NC State',
        'Miami (FL)': 'Miami',
        'Pittsburgh': 'Pitt',
        'Florida St.': 'Florida State',
    }

    return mappings.get(haslametrics_name, haslametrics_name)


def merge_with_baseline(
    haslametrics_df: pd.DataFrame,
    baseline_df: pd.DataFrame,
    on: str = 'team'
) -> pd.DataFrame:
    """
    Merge Haslametrics data with baseline team stats.

    Args:
        haslametrics_df: DataFrame from Haslametrics
        baseline_df: Baseline team stats (e.g., from Barttorvik)
        on: Column to merge on (typically 'team')

    Returns:
        Merged DataFrame
    """
    # Standardize team names in Haslametrics data
    if 'team' in haslametrics_df.columns:
        haslametrics_df['team'] = haslametrics_df['team'].apply(standardize_team_name)

    # Merge
    merged = baseline_df.merge(
        haslametrics_df,
        on=on,
        how='left',
        suffixes=('', '_hasla')
    )

    logger.info(f"Merged {len(merged)} teams with Haslametrics data")
    return merged


def clean_haslametrics_ratings(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean Haslametrics ratings DataFrame by removing multi-level headers.

    Args:
        df: Raw DataFrame from Haslametrics HTML table

    Returns:
        Cleaned DataFrame with proper column names
    """
    # Find the row with actual column headers (contains 'Rk' and 'Team')
    # Skip blank rows and multi-level header rows
    for idx in range(min(10, len(df))):
        row = df.iloc[idx]
        if 'Rk' in str(row.iloc[0]) or (pd.notna(row.iloc[0]) and str(row.iloc[0]).strip() == 'Rk'):
            # This is the header row
            new_columns = df.iloc[idx].tolist()
            df = df.iloc[idx+1:].copy()
            df.columns = new_columns
            break

    # Remove blank rows
    df = df[df.iloc[:, 0].notna()].copy()

    # Reset index
    df = df.reset_index(drop=True)

    logger.info(f"Cleaned Haslametrics data: {len(df)} teams, {len(df.columns)} columns")
    return df


def save_haslametrics_data(df: pd.DataFrame, filename: str, clean: bool = True):
    """
    Save Haslametrics data to directory.

    Args:
        df: DataFrame to save
        filename: Output filename
        clean: Whether to clean multi-level headers (default: True)
    """
    if clean and len(df) > 0:
        df = clean_haslametrics_ratings(df)

    output_path = HASLAMETRICS_DATA_DIR / filename
    df.to_csv(output_path, index=False)
    logger.info(f"Saved {len(df)} records to {output_path}")


if __name__ == '__main__':
    # Simple test
    print("Haslametrics Scraper Module")
    print(f"Base URL: {BASE_URL}")
    print(f"Rate limit: {HASLAMETRICS_CONFIG['rate_limit']} req/min")

    # Test ratings fetch
    print("\nFetching team ratings for 2026...")
    ratings = fetch_team_ratings(2026)
    print(f"Found {len(ratings)} teams")
    if len(ratings) > 0:
        print("\nColumns:", ratings.columns.tolist())
        print("\nFirst few rows:")
        print(ratings.head())
