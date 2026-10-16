import sys
import os
sys.path.append(os.path.abspath('.'))
from src.draft_generator import generate_predictions
from src.preprocessing import load_depth_charts

df = generate_predictions()
depth_map = load_depth_charts(2026)

print("=== Top WR2s in the 2026 Projections ===")
wr2s = []
wr1s = []

for _, row in df.iterrows():
    if row['position'] == 'WR':
        p_name = row['player_name']
        # The depth map expects clean_name which might be slightly different, but let's approximate
        pos_rank = depth_map.get((p_name.lower().replace('.', '').replace('-', '').strip(), 'WR'), 99)
        if pos_rank == 2:
            wr2s.append(row)
        elif pos_rank == 1:
            wr1s.append(row)

import pandas as pd
wr2_df = pd.DataFrame(wr2s).sort_values('predicted_pts', ascending=False)
wr1_df = pd.DataFrame(wr1s).sort_values('predicted_pts', ascending=False)

print("\\nTop 5 WR2s by Projected Points:")
print(wr2_df[['player_name', 'team_predict', 'predicted_pts', 'vorp', 'predicted_adp']].head(5).to_string(index=False))

print("\\nTop 5 WR1s by Projected Points (For Comparison):")
print(wr1_df[['player_name', 'team_predict', 'predicted_pts', 'vorp', 'predicted_adp']].head(5).to_string(index=False))
