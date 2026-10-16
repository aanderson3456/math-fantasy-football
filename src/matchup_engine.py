import os
import re
import json
import math
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
CACHE_DIR = os.path.join(DATA_DIR, 'cache')
PREDS_PATH = os.path.join(CACHE_DIR, 'live_12team_predictions.parquet')
ROSTERS_PATH = os.path.join(DATA_DIR, 'rosters.json')
SETTINGS_PATH = os.path.join(DATA_DIR, 'settings.json')
from src.config import LEAGUE_CONFIG
league_matchups_path = os.path.join(CACHE_DIR, f'yahoo_week1_all_matchups_{LEAGUE_CONFIG}.json')
YAHOO_MATCHUPS_PATH = league_matchups_path if os.path.exists(league_matchups_path) else os.path.join(CACHE_DIR, 'yahoo_week1_all_matchups.json')

def normalize_name(name):
    name = str(name).lower()
    name = re.sub(r'[\.\',\-]', '', name)
    name = re.sub(r'\b(jr|sr|iii|ii|iv|v)\b', '', name)
    name = re.sub(r'\s+', ' ', name).strip()
    return name

class MatchupEngine:
    def __init__(self):
        self.preds_df = pd.read_parquet(PREDS_PATH) if os.path.exists(PREDS_PATH) else pd.DataFrame()
        self.nn_lookup = self._build_nn_lookup()
        self.settings = self._load_settings()
        self.exact_yahoo_matchups = self._load_exact_yahoo_matchups()

    def _load_settings(self):
        if os.path.exists(SETTINGS_PATH):
            with open(SETTINGS_PATH, 'r') as f:
                return json.load(f)
        return {"league_teams": [], "my_team": "Pitts=tits. Shedeur=ass💦💦"}

    def _load_exact_yahoo_matchups(self):
        if os.path.exists(YAHOO_MATCHUPS_PATH):
            with open(YAHOO_MATCHUPS_PATH, 'r') as f:
                return json.load(f)
        return []

    def _build_nn_lookup(self):
        lookup = {}
        if len(self.preds_df) == 0:
            return lookup

        for _, row in self.preds_df.iterrows():
            pname = row['player_name']
            norm = normalize_name(pname)
            
            # Position-specific weekly scoring scale to align with league scoring
            pos = str(row['position']).upper()
            raw_season_pts = float(row['predicted_pts'])
            
            if pos == 'QB':
                # Standard 4-point passing TD scoring (17-game NFL schedule)
                weekly_pts = round(raw_season_pts / 17.0, 2)
            elif pos == 'DEF':
                weekly_pts = round(raw_season_pts / 17.0, 2)
            elif pos in ['RB', 'WR']:
                weekly_pts = round((raw_season_pts / 16.0) * 1.05, 2)
            else:
                weekly_pts = round(raw_season_pts / 17.0, 2)
                
            weekly_std = round(weekly_pts * 0.28, 2)
            
            reasons = []
            if row.get('is_young_innovator') == 1.0:
                reasons.append("Modern High-Tempo Scheme")
            if row.get('oline_score', 0) > 0.5:
                reasons.append("Top-Tier O-Line Protection")
            elif row.get('oline_score', 0) < -0.5:
                reasons.append("Vulnerable O-Line Matchup")
            if row.get('coach_bellcow_index', 0.5) > 0.65:
                reasons.append("Concentrated Bellcow Touch Volume")
            if row.get('coach_pass_rate', 0.5) > 0.58:
                reasons.append("Pass-Heavy Game Script Floor")
            if row.get('years_exp', 1) <= 2:
                reasons.append("Ascending Young Profile")

            lookup[norm] = {
                'player_name': pname,
                'pos': pos,
                'team': str(row.get('team_predict', 'UNK')),
                'nn_proj': weekly_pts,
                'nn_std': weekly_std,
                'vorp': round(float(row['vorp']), 1),
                'reasons': reasons or ["Standard Volume Projection"]
            }
        return lookup

    def enrich_player(self, p_dict):
        if 'player' in p_dict and isinstance(p_dict['player'], dict):
            slot = p_dict.get('slot', p_dict['player'].get('slot', 'BN'))
            p_dict = {**p_dict['player'], 'slot': slot}
        pname = p_dict.get('player_name', 'Unknown')
        norm = normalize_name(pname)
        
        nn_item = self.nn_lookup.get(norm)
        if not nn_item:
            for k, v in self.nn_lookup.items():
                if len(k) >= 5 and (k in norm or norm in k):
                    nn_item = v
                    break

        y_pts = float(p_dict.get('yahoo_proj', 0.0))
        y_min = round(max(0.0, y_pts * 0.65), 2)
        y_max = round(y_pts * 1.35, 2)

        if nn_item:
            nn_pts = nn_item['nn_proj']
            nn_std = nn_item['nn_std']
            pos = nn_item['pos'] if p_dict.get('pos') in ['UNK', ''] else p_dict.get('pos', nn_item['pos'])
            team = p_dict.get('team') if p_dict.get('team') not in ['UNK', ''] else nn_item['team']
            reasons = nn_item['reasons']
        else:
            # Baseline fallback
            nn_pts = y_pts
            nn_std = round(y_pts * 0.3, 2)
            pos = p_dict.get('pos', 'WR')
            team = p_dict.get('team', 'UNK')
            reasons = ["Baseline Projection"]

        diff = round(nn_pts - y_pts, 2)
        if diff >= 1.5:
            div_type = "bullish"
            div_label = f"+{diff:.1f} NN Bullish"
        elif diff <= -1.5:
            div_type = "bearish"
            div_label = f"{diff:.1f} NN Bearish"
        else:
            div_type = "neutral"
            div_label = "Agreement"

        return {
            'player_name': pname,
            'pos': pos,
            'team': team,
            'opp': p_dict.get('opp', ''),
            'yahoo_proj': y_pts,
            'yahoo_min': y_min,
            'yahoo_max': y_max,
            'nn_proj': nn_pts,
            'nn_std': nn_std,
            'diff': diff,
            'divergence_type': div_type,
            'divergence_label': div_label,
            'reasons': reasons
        }

    def compute_matchup_projections(self, team1_starters, team2_starters):
        t1_nn_pts = sum(s['player']['nn_proj'] for s in team1_starters)
        t1_nn_var = sum((s['player']['nn_std']) ** 2 for s in team1_starters)
        t1_yahoo_pts = sum(s['player']['yahoo_proj'] for s in team1_starters)

        t2_nn_pts = sum(s['player']['nn_proj'] for s in team2_starters)
        t2_nn_var = sum((s['player']['nn_std']) ** 2 for s in team2_starters)
        t2_yahoo_pts = sum(s['player']['yahoo_proj'] for s in team2_starters)

        # NN Win Probability (Gaussian CDF)
        diff_mu_nn = t1_nn_pts - t2_nn_pts
        diff_sigma_nn = math.sqrt(max(1.0, t1_nn_var + t2_nn_var))
        z_nn = diff_mu_nn / diff_sigma_nn
        nn_win_prob_t1 = round((1.0 + math.erf(z_nn / math.sqrt(2.0))) / 2.0 * 100.0, 1)
        nn_win_prob_t2 = round(100.0 - nn_win_prob_t1, 1)

        # Yahoo Win Probability (Exact point differential scale)
        diff_mu_y = t1_yahoo_pts - t2_yahoo_pts
        z_y = diff_mu_y / 22.0
        yahoo_win_prob_t1 = round((1.0 + math.erf(z_y / math.sqrt(2.0))) / 2.0 * 100.0, 1)
        yahoo_win_prob_t2 = round(100.0 - yahoo_win_prob_t1, 1)

        nn_fav = 1 if t1_nn_pts >= t2_nn_pts else 2
        y_fav = 1 if t1_yahoo_pts >= t2_yahoo_pts else 2
        models_agree = (nn_fav == y_fav)

        return {
            'team1': {
                'nn_pts': round(t1_nn_pts, 2),
                'yahoo_pts': round(t1_yahoo_pts, 2),
                'nn_win_prob': nn_win_prob_t1,
                'yahoo_win_prob': yahoo_win_prob_t1
            },
            'team2': {
                'nn_pts': round(t2_nn_pts, 2),
                'yahoo_pts': round(t2_yahoo_pts, 2),
                'nn_win_prob': nn_win_prob_t2,
                'yahoo_win_prob': yahoo_win_prob_t2
            },
            'models_agree': models_agree,
            'divergence_summary': "Models Agree on Projected Winner" if models_agree else "⚡ Model Disagreement: NN & Yahoo Pick Different Winners!"
        }

    def get_full_matchup_payload(self, matchup_index=0):
        if matchup_index >= len(self.exact_yahoo_matchups):
            matchup_index = 0

        raw_matchup = self.exact_yahoo_matchups[matchup_index]
        t1_raw = raw_matchup['team1']
        t2_raw = raw_matchup['team2']

        t1_starters = [{'slot': s['slot'], 'player': self.enrich_player(s)} for s in t1_raw['starters']]
        t1_bench = [{'slot': 'BN', 'player': self.enrich_player(b)} for b in t1_raw['bench']]

        t2_starters = [{'slot': s['slot'], 'player': self.enrich_player(s)} for s in t2_raw['starters']]
        t2_bench = [{'slot': 'BN', 'player': self.enrich_player(b)} for b in t2_raw['bench']]

        summary = self.compute_matchup_projections(t1_starters, t2_starters)

        matchup_list = []
        for idx, m in enumerate(self.exact_yahoo_matchups):
            matchup_list.append({
                'index': idx,
                'team1_name': m['team1']['name'],
                'team2_name': m['team2']['name'],
                'is_my_matchup': (m['team1']['name'] == self.settings.get('my_team') or m['team2']['name'] == self.settings.get('my_team'))
            })

        return {
            'week': 1,
            'matchup_index': matchup_index,
            'matchups_list': matchup_list,
            'team1': {
                'name': t1_raw['name'],
                'is_my_team': (t1_raw['name'] == self.settings.get('my_team')),
                'starters': t1_starters,
                'bench': t1_bench,
                'metrics': summary['team1']
            },
            'team2': {
                'name': t2_raw['name'],
                'is_my_team': (t2_raw['name'] == self.settings.get('my_team')),
                'starters': t2_starters,
                'bench': t2_bench,
                'metrics': summary['team2']
            },
            'summary': summary
        }

if __name__ == '__main__':
    engine = MatchupEngine()
    payload = engine.get_full_matchup_payload(0)
    print(f"Week 1 Matchup: {payload['team1']['name']} vs {payload['team2']['name']}")
    print(f"Yahoo Score: {payload['summary']['team1']['yahoo_pts']} pts ({payload['summary']['team1']['yahoo_win_prob']}%) vs {payload['summary']['team2']['yahoo_pts']} pts ({payload['summary']['team2']['yahoo_win_prob']}%)")
    print(f"NN Score:    {payload['summary']['team1']['nn_pts']} pts ({payload['summary']['team1']['nn_win_prob']}%) vs {payload['summary']['team2']['nn_pts']} pts ({payload['summary']['team2']['nn_win_prob']}%)")
    for s in payload['team1']['starters']:
        p = s['player']
        print(f"  {s['slot']:<5} {p['player_name']:<25} | Yahoo: {p['yahoo_proj']:>5.2f} pts | NN: {p['nn_proj']:>5.2f} pts | {p['divergence_label']}")
