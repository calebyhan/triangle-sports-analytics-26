"""
Enhanced Selenium scraper for Haslametrics that parses DOM directly.
"""

import pandas as pd
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
import time
from ..logger import setup_logger

logger = setup_logger(__name__)

# Column headers for Haslametrics ratings table (offensive summary section)
HASLAMETRICS_COLUMNS = [
    'Rk', 'Team', 'Eff', 'FTAR', 'FT%', 'FGAR', 'FG%', '3PAR', '3P%', 'MRAR',
    'MR%', 'NPAR', 'NP%', 'PPSt', 'PPSC', 'SCC%', '%3PA', '%MRA', '%NPA', 'Prox', 'AP%'
]


def get_headless_chrome():
    """Create a headless Chrome webdriver"""
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--window-size=1920,1080")

    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=chrome_options)
    return driver


def fetch_haslametrics_table(year: int = 2026, base_url: str = "https://www.haslametrics.com") -> pd.DataFrame:
    """
    Fetch Haslametrics ratings table by parsing DOM directly with Selenium.

    Args:
        year: Season year (e.g., 2026 for 2025-26 season)
        base_url: Haslametrics base URL

    Returns:
        DataFrame with team ratings
    """
    url = f"{base_url}/ratings.php?yr={year}"

    logger.info(f"Fetching Haslametrics via Selenium (DOM extraction) for year {year}...")

    driver = None
    try:
        driver = get_headless_chrome()
        driver.get(url)

        # Wait for table to load
        wait = WebDriverWait(driver, 20)
        table = wait.until(EC.presence_of_element_located((By.ID, "myTable")))

        # Give JavaScript time to populate cells
        time.sleep(8)

        # Extract all rows from the table
        all_rows = table.find_elements(By.TAG_NAME, "tr")

        logger.info(f"Found {len(all_rows)} total rows in table")

        data = []
        for i, row in enumerate(all_rows):
            cells = row.find_elements(By.TAG_NAME, "td")

            if cells:
                row_data = [cell.text.strip() for cell in cells]

                # Skip rows where first cell is empty or doesn't look like a rank (number)
                if row_data and row_data[0] and row_data[0].isdigit():
                    data.append(row_data)

        logger.info(f"Extracted {len(data)} data rows")

        # Create DataFrame with predefined columns
        df = pd.DataFrame(data)

        # Assign column names (use predefined or generate generic names)
        if len(df.columns) == len(HASLAMETRICS_COLUMNS):
            df.columns = HASLAMETRICS_COLUMNS
        else:
            logger.warning(f"Column mismatch: expected {len(HASLAMETRICS_COLUMNS)}, got {len(df.columns)}")
            df.columns = [f'col_{i}' for i in range(len(df.columns))]

        logger.info(f"Final DataFrame: {len(df)} teams with {len(df.columns)} columns")
        return df

    except Exception as e:
        logger.error(f"Failed to fetch Haslametrics with Selenium DOM extraction: {e}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame()

    finally:
        if driver:
            driver.quit()


if __name__ == "__main__":
    # Test
    df = fetch_haslametrics_table(2026)
    print(f"Shape: {df.shape}")
    if len(df) > 0:
        print(f"\nColumns: {df.columns.tolist()}")
        print(f"\nFirst 5 rows:")
        print(df.head())
        print(f"\nSample teams:")
        print(df[df['Team'].str.contains('Duke|North Carolina', na=False)][['Rk', 'Team', 'Eff']].head())
