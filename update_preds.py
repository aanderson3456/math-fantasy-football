import sys
import os
sys.path.append(os.path.abspath('.'))
os.environ['LEAGUE_CONFIG'] = 'mafia'
from app import generate_predictions
df = generate_predictions()
df.to_parquet('data/cache/live_12team_predictions.parquet')
print("Updated predictions parquet!")
