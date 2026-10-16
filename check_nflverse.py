import sys
try:
    import nfl_data_py as nfl
    print("nfl_data_py version:", nfl.__version__)
    
    # Check if there is a depth chart function
    methods = [m for m in dir(nfl) if 'depth' in m.lower() or 'roster' in m.lower()]
    print("Depth/Roster methods in nfl_data_py:", methods)
    
except ImportError:
    print("nfl_data_py not installed in this env")

# We can also check nflverse-data github releases via pandas
import pandas as pd
print("\\nChecking if we can download nflverse depth charts directly...")
try:
    # URL for depth charts in nflverse
    url = "https://github.com/nflverse/nflverse-data/releases/download/depth_charts/depth_charts_2025.parquet"
    df = pd.read_parquet(url)
    print("Successfully loaded 2025 depth charts from nflverse!")
    print("Columns:", df.columns.tolist())
    print(df.head(2))
except Exception as e:
    print("Error downloading 2025 depth charts:", e)

try:
    # URL for depth charts in nflverse (maybe it's without the year)
    url = "https://github.com/nflverse/nflverse-data/releases/download/depth_charts/depth_charts.parquet"
    df = pd.read_parquet(url)
    print("Successfully loaded ALL depth charts from nflverse!")
    print("Columns:", df.columns.tolist())
    print("Years available:", df['season'].unique() if 'season' in df.columns else 'Unknown')
except Exception as e:
    print("Error downloading ALL depth charts:", e)
