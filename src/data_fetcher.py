import os
import urllib.request
import json
import pandas as pd
from src.config import START_YEAR, END_YEAR, PREDICT_YEAR, CACHE_DIR

# FFC API endpoints in priority order
ADP_FORMATS = ['half-ppr', 'ppr', 'standard']

def download_file(url, filepath):
    """Downloads a file from a URL and saves it to the specified path."""
    if os.path.exists(filepath):
        print(f"File already exists: {filepath}")
        return
    print(f"Downloading {url} to {filepath}...")
    try:
        # Set User-Agent to avoid getting blocked
        req = urllib.request.Request(
            url, 
            headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
        )
        with urllib.request.urlopen(req) as response:
            with open(filepath, 'wb') as f:
                f.write(response.read())
        print("Download complete.")
    except Exception as e:
        print(f"Error downloading {url}: {e}")

def fetch_player_stats():
    """Downloads weekly player stats from stats_player tag for each year, combines them, and renames columns."""
    filepath = os.path.join(CACHE_DIR, "player_stats.parquet")
    if os.path.exists(filepath):
        print(f"Combined player stats file already exists: {filepath}")
        return filepath
        
    print("Fetching and combining seasonal player stats...")
    dfs = []
    # Loop from START_YEAR to PREDICT_YEAR - 1 (which includes 2025)
    for year in range(START_YEAR, PREDICT_YEAR):
        url = f"https://github.com/nflverse/nflverse-data/releases/download/stats_player/stats_player_week_{year}.parquet"
        temp_path = os.path.join(CACHE_DIR, f"stats_player_week_{year}.parquet")
        download_file(url, temp_path)
        if os.path.exists(temp_path):
            df = pd.read_parquet(temp_path)
            dfs.append(df)
            # Remove temp file to save space
            os.remove(temp_path)
            
    if not dfs:
        raise RuntimeError("No player stats files were downloaded successfully!")
        
    combined_df = pd.concat(dfs, ignore_index=True)
    
    # Rename columns to match pipeline expectations
    rename_dict = {
        'passing_interceptions': 'interceptions',
        'sacks_suffered': 'sacks',
        'team': 'recent_team',
        'sack_yards_lost': 'sack_yards'
    }
    combined_df.rename(columns=rename_dict, inplace=True)
    
    combined_df.to_parquet(filepath, index=False)
    print(f"Successfully created combined player stats at {filepath}")
    return filepath

def fetch_rosters():
    """Downloads rosters for each year in our historical range."""
    # We need rosters up to PREDICT_YEAR - 1 (since draft Y uses Y-1 stats)
    roster_paths = {}
    for year in range(START_YEAR - 1, PREDICT_YEAR):
        url = f"https://github.com/nflverse/nflverse-data/releases/download/rosters/roster_{year}.parquet"
        filepath = os.path.join(CACHE_DIR, f"roster_{year}.parquet")
        download_file(url, filepath)
        if os.path.exists(filepath):
            roster_paths[year] = filepath
    return roster_paths

def fetch_adp_data():
    """Fetches ADP data from Fantasy Football Calculator API for 2008 to END_YEAR."""
    adp_paths = {}
    # FFC has data from 2008 onwards
    start_adp_year = max(2008, START_YEAR)
    for year in range(start_adp_year, END_YEAR + 1):
        filepath = os.path.join(CACHE_DIR, f"adp_{year}.json")
        if os.path.exists(filepath):
            print(f"ADP data for {year} already cached at {filepath}")
            adp_paths[year] = filepath
            continue
            
        success = False
        for fmt in ADP_FORMATS:
            url = f"https://fantasyfootballcalculator.com/api/v1/adp/{fmt}?teams=12&year={year}"
            print(f"Attempting to fetch ADP for {year} in format '{fmt}' from {url}...")
            try:
                req = urllib.request.Request(
                    url, 
                    headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
                )
                with urllib.request.urlopen(req) as response:
                    data = json.loads(response.read().decode('utf-8'))
                    
                if data.get('status') == 'Success' and len(data.get('players', [])) > 0:
                    with open(filepath, 'w', encoding='utf-8') as f:
                        json.dump(data, f, indent=4)
                    print(f"Successfully cached ADP for {year} using format '{fmt}'")
                    adp_paths[year] = filepath
                    success = True
                    break
                else:
                    print(f"Format '{fmt}' failed for {year}: {data.get('errors', 'Unknown error')}")
            except Exception as e:
                print(f"Exception trying format '{fmt}' for {year}: {e}")
                
        if not success:
            print(f"WARNING: Could not fetch ADP for {year} in any format.")
            
    return adp_paths

def run_data_fetch():
    """Executes the full data collection pipeline."""
    print("Starting data collection pipeline...")
    stats_path = fetch_player_stats()
    roster_paths = fetch_rosters()
    adp_paths = fetch_adp_data()
    print("Data collection pipeline complete!")
    return stats_path, roster_paths, adp_paths

if __name__ == "__main__":
    run_data_fetch()
