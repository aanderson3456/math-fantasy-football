import pandas as pd
import urllib.request
import os
import json
import time
import re
import urllib.parse
import urllib.error

# Download games.parquet if we don't have it
if not os.path.exists('games.parquet'):
    url = 'https://github.com/nflverse/nflverse-data/releases/download/schedules/games.parquet'
    urllib.request.urlretrieve(url, 'games.parquet')

df = pd.read_parquet('games.parquet')
home_coaches = df['home_coach'].dropna().unique()
away_coaches = df['away_coach'].dropna().unique()
all_coaches = set(home_coaches).union(set(away_coaches))
# Remove nulls or empties
all_coaches = {c for c in all_coaches if c and isinstance(c, str)}

print(f"Found {len(all_coaches)} unique coaches.")

# Let's save the list so we can query Wikipedia
with open('coaches_list.json', 'w') as f:
    json.dump(list(all_coaches), f, indent=4)
