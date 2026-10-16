import pandas as pd
import urllib.request
url = 'https://github.com/nflverse/nflverse-data/releases/download/schedules/games.parquet'
urllib.request.urlretrieve(url, 'games.parquet')
df = pd.read_parquet('games.parquet')
print(df.columns.tolist())
