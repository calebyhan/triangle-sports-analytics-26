"""
Better parser for Haslametrics data with complex HTML tables.
"""

import pandas as pd
from bs4 import BeautifulSoup
import re


def parse_haslametrics_ratings(html: str) -> pd.DataFrame:
    """
    Parse Haslametrics ratings table, extracting rank, team, and efficiency.

    Args:
        html: Raw HTML from Haslametrics ratings page

    Returns:
        Cleaned DataFrame with team ratings
    """
    soup = BeautifulSoup(html, 'html.parser')

    # Find the main table
    table = soup.find('table', {'id': 'myTable'})
    if not table:
        table = soup.find('table')

    if not table:
        return pd.DataFrame()

    # Extract all rows
    rows = []
    for tr in table.find_all('tr'):
        cells = tr.find_all(['td', 'th'])
        if len(cells) >= 2:  # Must have at least rank and team
            row = [cell.get_text(strip=True) for cell in cells]
            rows.append(row)

    if len(rows) < 2:  # Need at least header + 1 data row
        return pd.DataFrame()

    # Use first row as headers or create generic headers
    headers = rows[0] if rows[0][0].lower() in ['rk', 'rank', ''] else None
    data_rows = rows[1:] if headers else rows

    # Filter out any remaining header rows in data
    data_rows = [row for row in data_rows if row[0].isdigit()]

    if not data_rows:
        return pd.DataFrame()

    # Create simplified DataFrame with just key columns
    # Column 0: Rank, Column 1: Team, Column 2: Eff (offensive efficiency)
    simplified_rows = []
    for row in data_rows:
        if len(row) >= 3:
            rank = row[0]
            team_raw = row[1]
            eff = row[2] if len(row) > 2 else ''

            # Extract team name (remove record in parentheses)
            team = re.sub(r'\s*\([^)]*\)', '', team_raw).strip()

            simplified_rows.append({
                'rank': rank,
                'team': team,
                'team_raw': team_raw,
                'offensive_efficiency': eff,
            })

    df = pd.DataFrame(simplified_rows)

    # Convert numeric columns
    df['rank'] = pd.to_numeric(df['rank'], errors='coerce')
    df['offensive_efficiency'] = pd.to_numeric(df['offensive_efficiency'], errors='coerce')

    # Remove any rows with missing data
    df = df[df['team'].notna() & (df['team'] != '')]
    df = df.reset_index(drop=True)

    return df


if __name__ == '__main__':
    print("Haslametrics Parser Module")
