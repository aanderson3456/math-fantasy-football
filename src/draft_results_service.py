import os
import re
import json
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
CACHE_DIR = os.path.join(DATA_DIR, 'cache')
PREDS_PATH = os.path.join(CACHE_DIR, 'live_12team_predictions.parquet')
ROSTERS_PATH = os.path.join(DATA_DIR, 'rosters.json')
SETTINGS_PATH = os.path.join(DATA_DIR, 'settings.json')
DRAFT_PICKS_PATH = os.path.join(CACHE_DIR, 'yahoo_full_draft_picks.json')
RAW_YAHOO_PATH = os.path.join(CACHE_DIR, 'yahoo_raw_projections_2026.txt')
GRADES_REPORT_PATH = os.path.join(CACHE_DIR, 'draft_grades_report.json')

def normalize_name(name):
    name = str(name).lower()
    name = re.sub(r'[\.\',\-]', '', name)
    name = re.sub(r'\b(jr|sr|iii|ii|iv|v)\b', '', name)
    name = re.sub(r'\s+', ' ', name).strip()
    return name

class DraftResultsService:
    def __init__(self):
        self.preds_df = pd.read_parquet(PREDS_PATH) if os.path.exists(PREDS_PATH) else pd.DataFrame()
        self.settings = self._load_settings()
        self.raw_draft_picks = self._load_draft_picks()
        self.yahoo_ranks = self._load_yahoo_ranks()
        self.nn_lookup = self._build_nn_lookup()
        self.enriched_draft_log = self._enrich_draft_picks()
        self.team_grades = self._compute_team_grades()

    def _load_settings(self):
        if os.path.exists(SETTINGS_PATH):
            with open(SETTINGS_PATH, 'r') as f:
                return json.load(f)
        return {"league_teams": [], "my_team": "Pitts=tits. Shedeur=ass💦💦"}

    def _load_draft_picks(self):
        if os.path.exists(DRAFT_PICKS_PATH):
            with open(DRAFT_PICKS_PATH, 'r') as f:
                return json.load(f)
        return []

    def _load_yahoo_ranks(self):
        y_ranks = {}
        if not os.path.exists(RAW_YAHOO_PATH):
            return y_ranks

        with open(RAW_YAHOO_PATH, 'r') as f:
            lines = [l.strip() for l in f if l.strip()]

        idx = 0
        while idx < len(lines):
            line = lines[idx]
            if '\t' in line and re.search(r'\d+\.\d+', line):
                parts = line.split('\t')
                pname = lines[max(0, idx - 6)] if idx >= 6 else "Unknown"
                if pname in ['Q', 'D']:
                    pname = lines[max(0, idx - 7)]
                try:
                    # Tab index 0 is rank / ADP, index 1 is avg pick, index 3 is proj
                    y_rank = float(parts[0])
                    s_proj = float(parts[3])
                    norm = normalize_name(pname)
                    y_ranks[norm] = {
                        'yahoo_adp': y_rank,
                        'yahoo_proj_season': s_proj
                    }
                except Exception:
                    pass
            idx += 1
        return y_ranks

    def _build_nn_lookup(self):
        lookup = {}
        if len(self.preds_df) == 0:
            return lookup

        # Sort by predicted points to get model rank
        sorted_df = self.preds_df.sort_values(by='predicted_pts', ascending=False).reset_index(drop=True)
        for rank_idx, row in sorted_df.iterrows():
            pname = row['player_name']
            norm = normalize_name(pname)
            pts = float(row['predicted_pts'])
            vorp = float(row['vorp'])
            pred_adp = float(row.get('predicted_adp', rank_idx + 1))
            pos = str(row['position'])
            team = str(row.get('team_predict', 'UNK'))

            lookup[norm] = {
                'player_name': pname,
                'position': pos,
                'team': team,
                'nn_proj_pts': round(pts, 1),
                'nn_vorp': round(vorp, 1),
                'nn_rank': rank_idx + 1,
                'nn_adp': round(pred_adp, 1)
            }
        return lookup

    def _enrich_draft_picks(self):
        enriched = []
        teams = self.settings.get('league_teams', [])
        
        for pick in self.raw_draft_picks:
            overall_pick = pick.get('overall_pick', len(enriched) + 1)
            pname = pick.get('player_name', '')
            norm = normalize_name(pname)
            mgr = pick.get('manager', 'Unknown')

            # Find matching full team name
            full_team_name = mgr
            for t in teams:
                if t.startswith(mgr.replace('...', '').strip()) or mgr.startswith(t[:8]):
                    full_team_name = t
                    break

            # NN Lookup
            nn_info = self.nn_lookup.get(norm)
            if not nn_info:
                for k, v in self.nn_lookup.items():
                    if len(k) >= 5 and (k in norm or norm in k):
                        nn_info = v
                        break

            # Yahoo Lookup
            y_info = self.yahoo_ranks.get(norm)
            if not y_info:
                for k, v in self.yahoo_ranks.items():
                    if len(k) >= 5 and (k in norm or norm in k):
                        y_info = v
                        break

            if nn_info:
                nn_pts = nn_info['nn_proj_pts']
                nn_vorp = nn_info['nn_vorp']
                nn_adp = nn_info['nn_adp']
                pos = nn_info['position']
                nfl_team = nn_info['team']
            else:
                nn_pts = 110.0
                nn_vorp = 10.0
                nn_adp = 100.0
                pos = pick.get('position', 'WR')
                nfl_team = pick.get('nfl_team', 'UNK')

            y_adp = y_info['yahoo_adp'] if y_info else round(overall_pick * 0.95, 1)
            y_pts = y_info['yahoo_proj_season'] if y_info else round(nn_pts * 0.98, 1)

            # Value differential: if model had him at ADP 20, but drafted at pick 48 -> +28 Steal!
            val_diff = round(overall_pick - nn_adp, 1)
            if val_diff >= 12.0:
                val_type = "steal"
                val_badge = f"+{val_diff:.0f} STEAL"
            elif val_diff <= -12.0:
                val_type = "reach"
                val_badge = f"{val_diff:.0f} REACH"
            else:
                val_type = "fair"
                val_badge = "FAIR VALUE"

            enriched.append({
                'overall_pick': overall_pick,
                'round': pick.get('round', ((overall_pick - 1) // 12) + 1),
                'pick_in_round': pick.get('pick_in_round', ((overall_pick - 1) % 12) + 1),
                'player_name': pname,
                'position': pos,
                'nfl_team': nfl_team,
                'fantasy_team': full_team_name,
                'manager': mgr,
                'nn_proj_pts': nn_pts,
                'nn_vorp': nn_vorp,
                'nn_adp': nn_adp,
                'yahoo_adp': y_adp,
                'yahoo_proj_pts': y_pts,
                'val_diff': val_diff,
                'val_type': val_type,
                'val_badge': val_badge,
                'is_my_pick': (full_team_name == self.settings.get('my_team'))
            })
        return enriched

    def _compute_team_grades(self):
        team_stats = {}
        for p in self.enriched_draft_log:
            tname = p['fantasy_team']
            if tname not in team_stats:
                team_stats[tname] = {
                    'team_name': tname,
                    'is_my_team': (tname == self.settings.get('my_team')),
                    'total_vorp': 0.0,
                    'total_pts': 0.0,
                    'picks': []
                }
            team_stats[tname]['total_vorp'] += p['nn_vorp']
            team_stats[tname]['total_pts'] += p['nn_proj_pts']
            team_stats[tname]['picks'].append(p)

        team_list = list(team_stats.values())
        if not team_list:
            return []

        all_vorps = [t['total_vorp'] for t in team_list]
        mean_v = np.mean(all_vorps)
        std_v = np.std(all_vorps) if np.std(all_vorps) > 0 else 1.0

        for t in team_list:
            z = (t['total_vorp'] - mean_v) / std_v
            if z >= 1.4: grade = "A+"
            elif z >= 0.9: grade = "A"
            elif z >= 0.4: grade = "A-"
            elif z >= 0.0: grade = "B+"
            elif z >= -0.4: grade = "B"
            elif z >= -0.8: grade = "B-"
            elif z >= -1.2: grade = "C+"
            elif z >= -1.6: grade = "C"
            else: grade = "C-"

            t['grade'] = grade
            t['total_vorp'] = round(t['total_vorp'], 1)
            t['total_pts'] = round(t['total_pts'], 1)

            # Steals and reaches
            sorted_picks = sorted(t['picks'], key=lambda x: x['val_diff'], reverse=True)
            t['best_steal'] = sorted_picks[0] if sorted_picks else None
            t['biggest_reach'] = sorted_picks[-1] if sorted_picks else None

        team_list.sort(key=lambda x: x['total_vorp'], reverse=True)
        for idx, t in enumerate(team_list, 1):
            t['rank'] = idx

        return team_list

    def get_full_payload(self):
        return {
            'draft_completed': True,
            'total_picks': len(self.enriched_draft_log),
            'num_rounds': 14,
            'my_team': self.settings.get('my_team', 'Pitts=tits. Shedeur=ass💦💦'),
            'team_grades': self.team_grades,
            'draft_log': self.enriched_draft_log
        }

if __name__ == '__main__':
    service = DraftResultsService()
    payload = service.get_full_payload()
    print(f"Loaded {payload['total_picks']} draft picks for {len(payload['team_grades'])} teams.")
    for t in payload['team_grades']:
        print(f"Rank {t['rank']:>2} | Grade: {t['grade']:<2} | VORP: {t['total_vorp']:>6.1f} | Pts: {t['total_pts']:>6.1f} | Team: {t['team_name']}")
        print(f"   Best Value: {t['best_steal']['player_name']} (Pick {t['best_steal']['overall_pick']}) | Reach: {t['biggest_reach']['player_name']} (Pick {t['biggest_reach']['overall_pick']})")
