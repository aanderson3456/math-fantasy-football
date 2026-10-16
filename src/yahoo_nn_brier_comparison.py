import os
import re
import json
import numpy as np
import pandas as pd
from scipy.stats import norm

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
YAHOO_LIST_PATH = os.path.join(BASE_DIR, 'data', 'cache', 'yahoo_live_draft_list.json')
NN_PREDS_PATH = os.path.join(BASE_DIR, 'data', 'cache', 'live_12team_predictions.parquet')
OUTPUT_PARQUET = os.path.join(BASE_DIR, 'data', 'cache', 'yahoo_vs_nn_draft_comparison.parquet')
OUTPUT_JSON = os.path.join(BASE_DIR, 'data', 'cache', 'yahoo_vs_nn_draft_comparison.json')

def normalize_name(name):
    name = str(name).lower()
    name = re.sub(r'[\.\',\-]', '', name)
    name = re.sub(r'\b(jr|sr|iii|ii|iv|v)\b', '', name)
    name = re.sub(r'\s+', ' ', name).strip()
    return name

def build_comparison():
    with open(YAHOO_LIST_PATH, 'r') as f:
        yahoo_players = json.load(f)

    nn_df = pd.read_parquet(NN_PREDS_PATH)

    # Build NN lookup
    nn_lookup = {}
    for idx, row in nn_df.iterrows():
        pname = row['player_name']
        norm_full = normalize_name(pname)
        parts = norm_full.split()
        initial_last = f"{parts[0][0]} {parts[-1]}" if len(parts) >= 2 else norm_full
        
        info = {
            'player_name': pname,
            'position': row['position'],
            'team': row.get('team_predict', 'UNK'),
            'nn_proj_pts': float(row['predicted_pts']),
            'nn_vorp': float(row.get('vorp', 0.0)),
            'nn_adp': float(row.get('predicted_adp', idx + 1)),
            'nn_rank': int(row.get('draft_rank', idx + 1))
        }
        nn_lookup[norm_full] = info
        nn_lookup[initial_last] = info

    # Specific manual alias overrides for Yahoo short names
    aliases = {
        'j cook iii': 'James Cook',
        'j cook': 'James Cook',
        'c mccaffrey': 'Christian McCaffrey',
        'm harrison jr': 'Marvin Harrison Jr.',
        'a st brown': 'Amon-Ra St. Brown',
        'j smith njigba': 'Jaxon Smith-Njigba',
        'j smithnjigba': 'Jaxon Smith-Njigba',
        'k walker iii': 'Kenneth Walker III',
        't etienne jr': 'Travis Etienne',
        'b thomas jr': 'Brian Thomas Jr.',
        'm nabers': 'Malik Nabers',
        'p nacua': 'Puka Nacua',
        'c lamb': 'CeeDee Lamb',
        'j jefferson': 'Justin Jefferson',
        'j gibbs': 'Jahmyr Gibbs',
        'b robinson': 'Bijan Robinson',
        'j chase': "Ja'Marr Chase",
        'j taylor': 'Jonathan Taylor',
        's barkley': 'Saquon Barkley',
        'd achane': "De'Von Achane",
        'c brown': 'Chase Brown',
        'k williams': 'Kyren Williams',
        't mcbride': 'Trey McBride',
        'b bowers': 'Brock Bowers',
        't higgins': 'Tee Higgins',
        'g pickens': 'George Pickens',
        'd london': 'Drake London',
        'z flowers': 'Zay Flowers',
        'c olave': 'Chris Olave',
        'n collins': 'Nico Collins',
        'j jacobs': 'Josh Jacobs',
        'b hall': 'Breece Hall',
        't mcmillan': 'Tetairoa McMillan',
        'j allen': 'Josh Allen',
        'j hurts': 'Jalen Hurts',
        'l jackson': 'Lamar Jackson',
        'j daniels': 'Jayden Daniels',
        'g wilson': 'Garrett Wilson',
        'k pitts': 'Kyle Pitts',
        'a jeanty': 'Ashton Jeanty',
        't henderson': 'TreVeyon Henderson'
    }

    matched = []
    for y in yahoo_players:
        y_name = y['name']
        norm_y = normalize_name(y_name)
        
        # Check alias first
        target_name = aliases.get(norm_y)
        nn_info = None
        if target_name:
            nn_info = nn_lookup.get(normalize_name(target_name))
            
        if not nn_info:
            nn_info = nn_lookup.get(norm_y)
            
        if not nn_info:
            # Fuzzy match
            for k, v in nn_lookup.items():
                if len(norm_y) >= 4 and (k.endswith(norm_y.split()[-1]) or norm_y in k or k in norm_y):
                    nn_info = v
                    break

        if nn_info:
            full_name = nn_info['player_name']
            y_proj = float(y.get('yahoo_proj_pts') or 0.0)
            nn_proj = nn_info['nn_proj_pts']
            pts_diff = round(nn_proj - y_proj, 2)
            
            y_xrank = float(y.get('yahoo_xrank') or 100.0)
            y_adp = float(y.get('yahoo_adp') or y_xrank)
            nn_rank = nn_info['nn_rank']
            rank_diff = round(y_xrank - nn_rank, 1) # Positive = NN ranks higher than Yahoo
            
            # Probabilistic Modeling:
            # Weekly boom event (weekly pts >= 15):
            y_wk_mean = y_proj / 16.0
            y_wk_std = max(1.0, y_wk_mean * 0.35)
            nn_wk_mean = nn_proj / 16.0
            nn_wk_std = max(1.0, nn_wk_mean * 0.35)
            
            p_boom_yahoo = 1.0 - norm.cdf(15.0, loc=y_wk_mean, scale=y_wk_std)
            p_boom_nn = 1.0 - norm.cdf(15.0, loc=nn_wk_mean, scale=nn_wk_std)
            
            # Elite season threshold (pts >= 250)
            p_elite_yahoo = 1.0 - norm.cdf(250.0, loc=y_proj, scale=max(10.0, y_proj * 0.25))
            p_elite_nn = 1.0 - norm.cdf(250.0, loc=nn_proj, scale=max(10.0, nn_proj * 0.25))
            
            # Brier Score Disagreement (Squared difference in predicted probabilities)
            brier_divergence_boom = float(round((p_boom_nn - p_boom_yahoo) ** 2, 4))
            brier_divergence_elite = float(round((p_elite_nn - p_elite_yahoo) ** 2, 4))
            
            # Stance classification
            if pts_diff > 25 or rank_diff > 10:
                verdict = "NN STRONG VALUE (STEAL)"
            elif pts_diff < -25 or rank_diff < -10:
                verdict = "NN FADE (OVERVALUED BY YAHOO)"
            else:
                verdict = "MODELS ALIGNED"

            matched.append({
                'player_name': full_name,
                'yahoo_name': y_name,
                'pos': nn_info['position'],
                'team': nn_info['team'],
                'yahoo_proj_pts': y_proj,
                'nn_proj_pts': round(nn_proj, 1),
                'pts_diff': pts_diff,
                'yahoo_xrank': y_xrank,
                'yahoo_adp': y_adp,
                'nn_rank': nn_rank,
                'rank_diff': rank_diff,
                'nn_vorp': round(nn_info['nn_vorp'], 1),
                'p_boom_yahoo': round(p_boom_yahoo, 3),
                'p_boom_nn': round(p_boom_nn, 3),
                'p_elite_yahoo': round(p_elite_yahoo, 3),
                'p_elite_nn': round(p_elite_nn, 3),
                'brier_divergence_boom': brier_divergence_boom,
                'brier_divergence_elite': brier_divergence_elite,
                'verdict': verdict
            })

    comp_df = pd.DataFrame(matched)
    comp_df.to_parquet(OUTPUT_PARQUET, index=False)
    with open(OUTPUT_JSON, 'w') as f:
        json.dump(matched, f, indent=2)

    return comp_df

if __name__ == '__main__':
    df = build_comparison()
    print(f"Successfully matched {len(df)} players between Yahoo and NN!")
    print("\n--- TOP 10 NN VALUES VS YAHOO (BIGGEST STEALS) ---")
    steals = df.sort_values(by='rank_diff', ascending=False).head(10)
    print(steals[['player_name', 'pos', 'team', 'yahoo_xrank', 'nn_rank', 'rank_diff', 'yahoo_proj_pts', 'nn_proj_pts', 'pts_diff', 'verdict']])
