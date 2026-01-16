"""
Shared utility functions for the Triangle Sports Analytics project.
"""
import ssl
import time
import urllib.request
import certifi
import pandas as pd
from io import StringIO
from urllib.error import URLError, HTTPError
from typing import Optional, Callable, Any


def fetch_url_with_retry(
    url: str,
    max_retries: int = 3,
    retry_delay: float = 1.0,
    timeout: int = 30,
    headers: Optional[dict] = None,
    parse_csv: bool = False
) -> Any:
    """
    Fetch a URL with exponential backoff retry logic and SSL verification.

    This function attempts to fetch a URL using proper SSL verification first.
    If all retries fail, it makes a final attempt without SSL verification
    as a fallback (with a warning).

    Args:
        url: URL to fetch
        max_retries: Maximum number of retry attempts with SSL verification
        retry_delay: Initial delay between retries in seconds (doubles each retry)
        timeout: Request timeout in seconds
        headers: Optional HTTP headers dict (default: Mozilla User-Agent)
        parse_csv: If True, parse response as CSV and return DataFrame

    Returns:
        If parse_csv=True: pandas DataFrame
        If parse_csv=False: decoded string content

    Raises:
        Exception: If all retries (including unverified SSL fallback) fail

    Example:
        >>> # Fetch and parse CSV
        >>> df = fetch_url_with_retry(
        ...     "https://example.com/data.csv",
        ...     parse_csv=True
        ... )

        >>> # Fetch HTML/text content
        >>> html = fetch_url_with_retry(
        ...     "https://example.com/page.html",
        ...     headers={'User-Agent': 'Custom Agent'}
        ... )
    """
    # Create SSL context with proper certificate validation
    ssl_context = ssl.create_default_context(cafile=certifi.where())

    # Set default headers if not provided
    if headers is None:
        headers = {'User-Agent': 'Mozilla/5.0'}

    req = urllib.request.Request(url, headers=headers)

    last_error = None

    # Attempt fetching with SSL verification
    for attempt in range(max_retries):
        try:
            with urllib.request.urlopen(req, context=ssl_context, timeout=timeout) as response:
                content = response.read().decode('utf-8')

                if parse_csv:
                    return pd.read_csv(StringIO(content))
                else:
                    return content

        except (URLError, HTTPError, ssl.SSLError) as e:
            last_error = e

            if attempt < max_retries - 1:
                # Not the last attempt - retry with exponential backoff
                wait_time = retry_delay * (2 ** attempt)
                print(f"   Attempt {attempt + 1}/{max_retries} failed: {e}")
                print(f"   Retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                # Last attempt with SSL failed - try without verification as fallback
                print(f"   ⚠ All {max_retries} attempts failed with SSL verification")
                print(f"   Final attempt without SSL verification...")

                try:
                    ssl_context_unverified = ssl.create_default_context()
                    ssl_context_unverified.check_hostname = False
                    ssl_context_unverified.verify_mode = ssl.CERT_NONE

                    with urllib.request.urlopen(req, context=ssl_context_unverified, timeout=timeout) as response:
                        content = response.read().decode('utf-8')

                        if parse_csv:
                            return pd.read_csv(StringIO(content))
                        else:
                            return content

                except Exception as fallback_error:
                    print(f"   ✗ Fallback also failed: {fallback_error}")
                    raise

        except Exception as e:
            # Unexpected error - don't retry
            print(f"   ✗ Unexpected error: {e}")
            raise

    # If we somehow get here, all retries failed
    raise last_error if last_error else RuntimeError(f"Failed to fetch URL: {url}")


def fetch_barttorvik_year(year: int, max_retries: int = 3, retry_delay: float = 1.0) -> pd.DataFrame:
    """
    Fetch team efficiency stats from Barttorvik for a specific year.

    This is a convenience wrapper around fetch_url_with_retry() specifically
    for Barttorvik data.

    Args:
        year: Season year to fetch (e.g., 2024 for 2023-24 season)
        max_retries: Maximum number of retry attempts
        retry_delay: Initial delay between retries (doubles each retry)

    Returns:
        DataFrame with team efficiency stats for the season

    Raises:
        Exception: If all retries failed

    Example:
        >>> df = fetch_barttorvik_year(2024)
        >>> print(df.columns)
        Index(['Team', 'Conf', 'G', 'Wins', 'Losses', 'AdjOE', 'AdjDE', ...])
    """
    url = f"https://barttorvik.com/{year}_team_results.csv"

    return fetch_url_with_retry(
        url=url,
        max_retries=max_retries,
        retry_delay=retry_delay,
        parse_csv=True
    )
