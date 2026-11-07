import os
import re
import sys
import json
import argparse
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)
DATA_DIR = os.path.join(BASE_DIR, 'data')
CACHE_DIR = os.path.join(DATA_DIR, 'cache')
BENCHMARK_DIR = os.path.join(DATA_DIR, 'benchmark')
PREDS_PATH = os.path.join(CACHE_DIR, 'live_12team_predictions.parquet')  # legacy fallback
HISTORY_PATH = os.path.join(BENCHMARK_DIR, 'prediction_history.parquet')
HISTORY_CSV = os.path.join(BENCHMARK_DIR, 'prediction_history.csv')

LEAGUES = ['mafia', '12_team_full_ppr']
GAMES_PER_SEASON = 17.0

os.makedirs(BENCHMARK_DIR, exist_ok=True)

def league_preds_path(league):
    return os.path.join(CACHE_DIR, f'nn_preds_{league}.parquet')

def default_league():
    lc = os.environ.get('LEAGUE_CONFIG', 'mafia')
    return lc if lc in LEAGUES else 'mafia'

def normalize_name(name):
    name = str(name).lower()
    name = re.sub(r'[\.\',\-]', '', name)
    name = re.sub(r'\b(jr|sr|iii|ii|iv|v)\b', '', name)
    name = re.sub(r'\s+', ' ', name).strip()
    return name

class BrierBenchmarkTracker:
    def __init__(self, season=2026):
        self.season = season
        self.history = self.load_history()

    def load_history(self):
        if os.path.exists(HISTORY_PATH):
            df = pd.read_parquet(HISTORY_PATH)
        elif os.path.exists(HISTORY_CSV):
            df = pd.read_csv(HISTORY_CSV)
        else:
            df = pd.DataFrame(columns=[
                'season', 'week', 'league', 'player_name', 'clean_name', 'position', 'team',
                'yahoo_proj_pts', 'yahoo_proj_min', 'yahoo_proj_max',
                'nn_proj_pts', 'nn_proj_std',
                'actual_pts', 'nn_abs_error', 'yahoo_abs_error', 'nn_sq_error', 'yahoo_sq_error',
                'nn_closer', 'brier_boom_nn', 'brier_boom_yahoo', 'is_scored'
            ])
        if 'league' not in df.columns:
            df['league'] = 'unknown'  # pre-league-column rows
        return df

    def save_history(self):
        self.history.to_parquet(HISTORY_PATH, index=False)
        self.history.to_csv(HISTORY_CSV, index=False)

    def _week_mask(self, week, league):
        return ((self.history['season'] == self.season) & (self.history['week'] == week)
                & (self.history['league'] == league))

    @staticmethod
    def _load_nn_predictions(league):
        path = league_preds_path(league)
        if os.path.exists(path):
            return pd.read_parquet(path)
        if os.path.exists(PREDS_PATH):
            print(f"Warning: {path} missing; falling back to {PREDS_PATH}. "
                  f"Run: LEAGUE_CONFIG={league} python src/generate_league_predictions.py")
            return pd.read_parquet(PREDS_PATH)
        raise FileNotFoundError("NN predictions not found. Run training/inference first.")

    def record_weekly_projections(self, week, yahoo_projections, nn_predictions_df=None, league=None):
        """
        Records/updates pre-game projections from both Yahoo and the Neural Network for all players.
        yahoo_projections: list of dicts [{'name': '...', 'pos': '...', 'team': '...', 'proj': 15.2, 'min': 8.0, 'max': 22.0}]
        league: league config key ('mafia' or '12_team_full_ppr'); NN projections come from that league's model.
        """
        league = league or default_league()
        if nn_predictions_df is None:
            nn_predictions_df = self._load_nn_predictions(league)

        from src.injury_service import InjuryAvailabilityService
        inj_service = InjuryAvailabilityService(season=self.season)

        # Build fast lookup for NN
        nn_lookup = {}
        for _, row in nn_predictions_df.iterrows():
            cname = normalize_name(row['player_name'])
            # Convert full season points to weekly projection (17-game season)
            weekly_nn = float(row['predicted_pts']) / GAMES_PER_SEASON
            weekly_std = weekly_nn * 0.35  # Standard deviation heuristic
            nn_lookup[cname] = {
                'player_name': row['player_name'],
                'position': str(row['position']),
                'team': str(row.get('team_predict', 'UNK')),
                'nn_proj_pts': weekly_nn,
                'nn_proj_std': weekly_std
            }

        records = []
        unmatched = []
        for y in yahoo_projections:
            pname = y.get('player_name') or y.get('name')
            cname = normalize_name(pname)
            pos = y.get('position') or y.get('pos', 'UNK')
            team = y.get('team', 'UNK')
            y_proj = float(y.get('proj_pts', y.get('proj', 0.0)))
            y_min = float(y.get('proj_min', y.get('min', max(0.0, y_proj * 0.5))))
            y_max = float(y.get('proj_max', y.get('max', y_proj * 1.5)))

            nn_info = nn_lookup.get(cname)
            if not nn_info and pos == 'DEF' and team not in ('UNK', None):
                # Yahoo names defenses by nickname ("Seahawks"); NN file uses "SEA DEF"
                nn_info = nn_lookup.get(normalize_name(f"{team} DEF"))
            if not nn_info:
                # Initial + last name matching (position must agree when known)
                parts = cname.split()
                if len(parts) >= 2:
                    init = parts[0][0]
                    last = parts[-1]
                    for k, v in nn_lookup.items():
                        k_parts = k.split()
                        if (len(k_parts) >= 2 and k_parts[-1] == last and k_parts[0][0] == init
                                and (pos in ('UNK', None) or v['position'] == pos)):
                            nn_info = v
                            break

            if not nn_info:
                # No NN projection for this player (e.g. K, rookies not in feature set).
                unmatched.append(pname)
                continue

            nn_proj = nn_info['nn_proj_pts']
            nn_std = nn_info['nn_proj_std']
            # NN file positions are authoritative; Yahoo-derived ones were historically regex-parsed
            pos = nn_info['position'] if nn_info['position'] not in ('', 'None', 'nan') else pos
            team = team if team not in ('UNK', None) else nn_info['team']

            mult = inj_service.get_availability_multiplier(team=team, week=week, status=y.get('status'))
            if y_proj == 0.0 and (mult == 0.0 or y.get('status') in ['IR', 'OUT', 'PUP', 'O']):
                nn_proj = 0.0
            else:
                nn_proj = nn_proj * mult

            records.append({
                'season': self.season,
                'week': week,
                'league': league,
                'player_name': pname,
                'clean_name': cname,
                'position': pos,
                'team': team,
                'yahoo_proj_pts': round(y_proj, 2),
                'yahoo_proj_min': round(y_min, 2),
                'yahoo_proj_max': round(y_max, 2),
                'nn_proj_pts': round(nn_proj, 2),
                'nn_proj_std': round(nn_std, 2),
                'actual_pts': np.nan,
                'nn_abs_error': np.nan,
                'yahoo_abs_error': np.nan,
                'nn_sq_error': np.nan,
                'yahoo_sq_error': np.nan,
                'nn_closer': np.nan,
                'brier_boom_nn': np.nan,
                'brier_boom_yahoo': np.nan,
                'is_scored': False
            })

        new_df = pd.DataFrame(records)
        if len(self.history) > 0:
            # Remove existing entries for this season, week and league
            filtered = self.history[~self._week_mask(week, league)]
            self.history = pd.concat([filtered, new_df], ignore_index=True)
        else:
            self.history = new_df

        self.save_history()
        print(f"[{league}] Recorded {len(new_df)} player projections for Week {week} (Season {self.season}); "
              f"{len(unmatched)} without an NN projection.")
        return len(new_df)

    def record_weekly_actuals(self, week, actuals_dict, league=None):
        """
        Updates actual fantasy points scored for week and computes error/Brier metrics.
        actuals_dict: {'player_name': actual_points}  (must be in the given league's scoring)
        """
        league = league or default_league()
        mask = self._week_mask(week, league)
        if not mask.any():
            print(f"[{league}] No projections found for Week {week}. Record projections first.")
            return

        # Normalization lookup for incoming actuals
        norm_actuals = {normalize_name(k): float(v) for k, v in actuals_dict.items()}
        self.history['nn_closer'] = self.history['nn_closer'].astype('object')
        self.history['is_scored'] = self.history['is_scored'].astype('object')

        for idx in self.history[mask].index:
            cname = self.history.at[idx, 'clean_name']
            act = norm_actuals.get(cname)
            if act is None:
                # Initial + last name matching
                parts = cname.split()
                if len(parts) >= 2:
                    init = parts[0][0]
                    last = parts[-1]
                    for k, v in norm_actuals.items():
                        k_parts = k.split()
                        if len(k_parts) >= 2 and k_parts[-1] == last and k_parts[0][0] == init:
                            act = v
                            break

            if act is not None:
                nn_p = self.history.at[idx, 'nn_proj_pts']
                y_p = self.history.at[idx, 'yahoo_proj_pts']

                nn_err = abs(nn_p - act)
                y_err = abs(y_p - act)
                nn_sq = (nn_p - act) ** 2
                y_sq = (y_p - act) ** 2

                # Brier Score on 'Boom' event (e.g. Scoring >= 15.0 pts)
                boom_threshold = 15.0
                actual_boom = 1.0 if act >= boom_threshold else 0.0
                
                # Probability estimate: Gaussian CDF proxy P(X >= 15)
                # For Yahoo: range proxy (Max - Min)
                y_range = max(1.0, self.history.at[idx, 'yahoo_proj_max'] - self.history.at[idx, 'yahoo_proj_min'])
                prob_boom_yahoo = np.clip((y_p - 5.0) / 20.0, 0.05, 0.95)
                
                # For NN: based on projected mean and std
                nn_std = max(1.0, self.history.at[idx, 'nn_proj_std'])
                from scipy.stats import norm as sp_norm
                prob_boom_nn = 1.0 - sp_norm.cdf(boom_threshold, loc=nn_p, scale=nn_std)

                brier_nn = (prob_boom_nn - actual_boom) ** 2
                brier_yahoo = (prob_boom_yahoo - actual_boom) ** 2

                self.history.at[idx, 'actual_pts'] = round(act, 2)
                self.history.at[idx, 'nn_abs_error'] = round(nn_err, 2)
                self.history.at[idx, 'yahoo_abs_error'] = round(y_err, 2)
                self.history.at[idx, 'nn_sq_error'] = round(nn_sq, 2)
                self.history.at[idx, 'yahoo_sq_error'] = round(y_sq, 2)
                self.history.at[idx, 'nn_closer'] = bool(nn_err < y_err)
                self.history.at[idx, 'brier_boom_nn'] = round(brier_nn, 4)
                self.history.at[idx, 'brier_boom_yahoo'] = round(brier_yahoo, 4)
                self.history.at[idx, 'is_scored'] = True

        self.save_history()
        print(f"[{league}] Updated actual scores and computed Brier metrics for Week {week}.")

    def ingest_matchup_file(self, week, league):
        """
        Records projections + actuals for a league/week from the cached Yahoo matchup dump
        (data/cache/yahoo_week{week}_all_matchups_{league}.json written by daily_yahoo_sync).
        Projections and actuals come from the same league, so scoring formats always match.
        """
        path = os.path.join(CACHE_DIR, f'yahoo_week{week}_all_matchups_{league}.json')
        if not os.path.exists(path):
            print(f"[{league}] {path} not found; run daily_yahoo_sync --week {week} first.")
            return False
        with open(path, 'r') as f:
            matchups = json.load(f)
        projs, actuals, seen = [], {}, set()
        for m in matchups:
            for side in ['team1', 'team2']:
                for p in m[side]['starters'] + m[side]['bench']:
                    n = p.get('player_name')
                    if not n:
                        continue
                    if 'actual_pts' in p:
                        actuals[n] = p['actual_pts']
                    if n not in seen:
                        seen.add(n)
                        projs.append({'player_name': n, 'pos': p.get('pos', 'UNK'), 'team': p.get('team', 'UNK'),
                                      'proj_pts': p.get('yahoo_proj', 0.0), 'status': p.get('status')})
        if not projs:
            print(f"[{league}] No players in {path}.")
            return False
        self.record_weekly_projections(week, projs, league=league)
        if actuals:
            self.record_weekly_actuals(week, actuals, league=league)
        return True

    def generate_evaluation_report(self, week=None, league=None):
        """
        Generates the comprehensive accuracy & Brier score report (week-by-week and cumulative).
        league: restrict to one league config; None = all leagues combined.
        """
        df = self.history[self.history['is_scored'] == True].copy()
        if league is not None:
            df = df[df['league'] == league]
        if len(df) == 0:
            print("No scored game weeks recorded yet. Projections are queued.")
            return self.get_queued_projections_summary()

        if week is not None:
            df = df[df['week'] == week]
            if len(df) == 0:
                return {'season': self.season, 'week': week, 'league': league or 'all',
                        'status': f'No scored data for Week {week}'}

        total_players = len(df)
        nn_wins = df['nn_closer'].sum()
        yahoo_wins = total_players - nn_wins
        nn_win_rate = (nn_wins / total_players) * 100.0

        nn_mae = df['nn_abs_error'].mean()
        yahoo_mae = df['yahoo_abs_error'].mean()

        nn_rmse = np.sqrt(df['nn_sq_error'].mean())
        yahoo_rmse = np.sqrt(df['yahoo_sq_error'].mean())

        nn_brier = df['brier_boom_nn'].mean()
        yahoo_brier = df['brier_boom_yahoo'].mean()

        # Breakdown by position
        pos_breakdown = []
        for pos, group in df.groupby('position'):
            p_total = len(group)
            p_nn_wins = group['nn_closer'].sum()
            pos_breakdown.append({
                'pos': pos,
                'count': p_total,
                'nn_closer_pct': round((p_nn_wins / p_total) * 100.0, 1),
                'nn_mae': round(group['nn_abs_error'].mean(), 2),
                'yahoo_mae': round(group['yahoo_abs_error'].mean(), 2),
                'nn_brier': round(group['brier_boom_nn'].mean(), 4),
                'yahoo_brier': round(group['brier_boom_yahoo'].mean(), 4)
            })

        # Biggest wins for NN
        df['margin'] = df['yahoo_abs_error'] - df['nn_abs_error']
        biggest_nn_steals = df.sort_values(by='margin', ascending=False).head(5)[
            ['player_name', 'position', 'team', 'actual_pts', 'nn_proj_pts', 'yahoo_proj_pts', 'margin']
        ].to_dict('records')

        report = {
            'season': self.season,
            'week': week or 'Season Cumulative',
            'league': league or 'all',
            'total_evaluated_players': total_players,
            'nn_closer_count': int(nn_wins),
            'yahoo_closer_count': int(yahoo_wins),
            'nn_win_rate_pct': round(nn_win_rate, 1),
            'nn_mae': round(nn_mae, 2),
            'yahoo_mae': round(yahoo_mae, 2),
            'nn_rmse': round(nn_rmse, 2),
            'yahoo_rmse': round(yahoo_rmse, 2),
            'nn_brier_score': round(nn_brier, 4),
            'yahoo_brier_score': round(yahoo_brier, 4),
            'brier_edge': "Neural Network" if nn_brier < yahoo_brier else "Yahoo",
            'positional_breakdown': pos_breakdown,
            'biggest_nn_wins': biggest_nn_steals
        }
        return report

    def get_queued_projections_summary(self):
        queued = self.history.groupby(['season', 'week', 'league']).agg(
            total_players=('player_name', 'count'),
            avg_nn_proj=('nn_proj_pts', 'mean'),
            avg_yahoo_proj=('yahoo_proj_pts', 'mean')
        ).reset_index()
        return {
            'status': 'Projections recorded (Awaiting games kickoff)',
            'queued_weeks': queued.to_dict('records')
        }

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Brier Score & Prediction Accuracy Tracker")
    parser.add_argument('--init_week', type=int, help='Initialize weekly projections from saved Yahoo dump')
    parser.add_argument('--ingest_week', type=int, help='Record projections+actuals from cached Yahoo matchup JSON')
    parser.add_argument('--league', choices=LEAGUES, help='League config (default: $LEAGUE_CONFIG or mafia)')
    parser.add_argument('--report', action='store_true', help='Generate accuracy and Brier score report')
    parser.add_argument('--week', type=int, help='Filter report by week')
    args = parser.parse_args()

    tracker = BrierBenchmarkTracker()

    if args.ingest_week:
        for lg in ([args.league] if args.league else LEAGUES):
            tracker.ingest_matchup_file(args.ingest_week, lg)

    if args.init_week:
        # Load from saved Yahoo dump
        from bs4 import BeautifulSoup
        raw_yahoo_path = os.path.join(CACHE_DIR, 'yahoo_raw_projections_2026.txt')
        if not os.path.exists(raw_yahoo_path):
            print(f"Error: {raw_yahoo_path} not found.")
            sys.exit(1)

        with open(raw_yahoo_path, 'r') as f:
            lines = [l.strip() for l in f if l.strip()]

        yahoo_projs = []
        idx = 0
        while idx < len(lines):
            line = lines[idx]
            # Match tab-separated stats line: e.g. "46.3\t58.9\t10\t214.60\t285.26\t144.23\t..."
            if '\t' in line and re.search(r'\d+\.\d+', line):
                parts = line.split('\t')
                # Walk backwards to find player name and position
                pname = lines[max(0, idx - 6)] if idx >= 6 else "Unknown"
                if pname == 'Q' or pname == 'D':
                    pname = lines[max(0, idx - 7)]
                pos = "WR"
                for back in range(1, 6):
                    if lines[idx - back] in ['QB', 'RB', 'WR', 'TE', 'K', 'DEF']:
                        pos = lines[idx - back]
                        pname = lines[idx - back - 1]
                        if pname == 'Q' or pname == 'D':
                            pname = lines[idx - back - 2]
                        break

                try:
                    # Season proj is around index 3
                    s_proj = float(parts[3])
                    s_max = float(parts[4]) if len(parts) > 4 else s_proj * 1.3
                    s_min = float(parts[5]) if len(parts) > 5 else s_proj * 0.7
                    # Convert to weekly rate
                    w_proj = s_proj / GAMES_PER_SEASON
                    w_max = s_max / GAMES_PER_SEASON
                    w_min = s_min / GAMES_PER_SEASON

                    yahoo_projs.append({
                        'player_name': pname,
                        'position': pos,
                        'proj_pts': w_proj,
                        'proj_min': w_min,
                        'proj_max': w_max
                    })
                except Exception:
                    pass
            idx += 1

        print(f"Parsed {len(yahoo_projs)} player projections from Yahoo dump.")
        tracker.record_weekly_projections(args.init_week, yahoo_projs, league=args.league)

    if args.report:
        rep = tracker.generate_evaluation_report(week=args.week, league=args.league)
        print(json.dumps(rep, indent=2, default=str))
