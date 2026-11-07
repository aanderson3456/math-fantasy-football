"""
Weekly two-branch stat-line network ("NN v2").

The season-long NN (app.py) predicts a season total from last year's stats, so its weekly
projection is just total / 17: no opponent, no Vegas line, no in-season form. This model
predicts a single game instead, and branches by league:

    features (form, usage, opponent, Vegas)
        -> shared trunk (MLP)
            -> stat-line branch: 12 expected box-score stats (src/scoring.STAT_COMPONENTS)
                -> mafia scoring leaf     -> mean points   (fixed dot product)
                -> 12-team scoring leaf   -> mean points   (fixed dot product)
            -> uncertainty branch: one learned sigma per league (heteroscedastic Gaussian)

Predicting stats instead of points means one network serves both leagues exactly, and any
future scoring change is a weight edit, not a retrain. The sigma heads give each projection
a spread, which is what the Brier scores in src/brier_report.py need.

Every feature for a game uses only games strictly before it (merge_asof, allow_exact_matches=False),
so evaluation on 2026 weeks with a model trained through 2025 is fully out of sample.

Usage:
    python src/weekly_model.py --train                    # train on 2006-2025, save
    python src/weekly_model.py --train --through 2026 8   # also fit on 2026 weeks 1-8
    python src/weekly_model.py --predict 2026 9           # write nn_weekly_preds_<league>_w9.parquet
"""
import os
import sys
import json
import argparse
import pickle

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.config import CACHE_DIR, standardize_team
from src.scoring import STAT_COMPONENTS, LEAGUES, harmonize_stats, league_points, scoring_vector

MODEL_DIR = os.path.join(CACHE_DIR, 'weekly_model')
POSITIONS = ['QB', 'RB', 'WR', 'TE']
USAGE = ['attempts', 'carries', 'targets']
TRACK = STAT_COMPONENTS + USAGE
DEF_PRIOR_GAMES = 4.0  # shrink opponent strength toward last season's value with this many pseudo-games


def normalize_name(name):
    import re
    name = str(name).lower()
    name = re.sub(r"[\.\',\-]", '', name)
    name = re.sub(r'\b(jr|sr|iii|ii|iv|v)\b', '', name)
    return re.sub(r'\s+', ' ', name).strip()


# ----------------------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------------------
def load_games_log():
    """All regular-season skill-position player-games, 2006 through the latest 2026 week."""
    frames = []
    for fname in ['player_stats.parquet', 'player_stats_2026.parquet']:
        p = os.path.join(CACHE_DIR, fname)
        if os.path.exists(p):
            frames.append(harmonize_stats(pd.read_parquet(p)))
    g = pd.concat(frames, ignore_index=True)
    g = g[(g['season_type'] == 'REG') & g['position'].isin(POSITIONS)].copy()
    g['name'] = g['player_display_name'].fillna(g['player_name']).fillna('Unknown')
    g['team'] = g['recent_team'].map(standardize_team)
    g['opp'] = g['opponent_team'].map(standardize_team)
    for c in USAGE:
        g[c] = g[c].fillna(0.0).astype(float)
    g['season'] = g['season'].astype('int64')
    g['week'] = g['week'].astype('int64')
    g['t'] = g['season'] * 100 + g['week']
    g = g.drop_duplicates(['player_id', 't']).sort_values(['player_id', 't']).reset_index(drop=True)
    g['pts_mafia'] = league_points(g, 'mafia')
    return g[['player_id', 'name', 'position', 'season', 'week', 't', 'team', 'opp'] + TRACK + ['pts_mafia']]


def load_vegas():
    """Team-perspective implied points and spread for every regular-season game."""
    s = pd.read_parquet(os.path.join(CACHE_DIR, 'games.parquet'))
    s = s[s['game_type'] == 'REG'].copy()
    s['season'] = s['season'].astype('int64')
    s['week'] = s['week'].astype('int64')
    rows = []
    for r in s.itertuples():
        total = r.total_line if pd.notna(r.total_line) else 44.0
        spread = r.spread_line if pd.notna(r.spread_line) else 0.0  # positive = home favored
        rows.append((r.season, r.week, standardize_team(r.home_team), standardize_team(r.away_team), 1.0, total / 2 + spread / 2, spread))
        rows.append((r.season, r.week, standardize_team(r.away_team), standardize_team(r.home_team), 0.0, total / 2 - spread / 2, -spread))
    return pd.DataFrame(rows, columns=['season', 'week', 'team', 'sched_opp', 'is_home', 'implied_total', 'team_spread'])


# ----------------------------------------------------------------------------------------
# Features
# ----------------------------------------------------------------------------------------
class FeatureBuilder:
    def __init__(self, games, vegas):
        self.g = games
        self.vegas = vegas
        self._build_player_state()
        self._build_defense_state()

    def _build_player_state(self):
        g = self.g
        st = g[['player_id', 'season', 'week', 't', 'team']].copy()
        grp = g.groupby('player_id')[TRACK]
        ewm = grp.transform(lambda s: s.ewm(halflife=3.0).mean())
        r8 = grp.transform(lambda s: s.rolling(8, min_periods=1).mean())
        sgrp = g.groupby(['player_id', 'season'])[TRACK]
        std = sgrp.cumsum().div(g.groupby(['player_id', 'season']).cumcount() + 1, axis=0)
        for c in TRACK:
            st[f'ewm_{c}'] = ewm[c].values
            st[f'r8_{c}'] = r8[c].values
            st[f'std_{c}'] = std[c].values
        st['n_season'] = g.groupby(['player_id', 'season']).cumcount() + 1
        st['n_career'] = g.groupby('player_id').cumcount() + 1
        self.player_state = st.rename(columns={'season': 'st_season', 'week': 'st_week', 'team': 'last_team'})

        prev = g.groupby(['player_id', 'season'])[TRACK].mean().reset_index()
        prev['season'] += 1
        self.prev_season = prev.rename(columns={c: f'prev_{c}' for c in TRACK})

    def _build_defense_state(self):
        g = self.g
        d = g.groupby(['opp', 'position', 'season', 'week', 't'])['pts_mafia'].sum().reset_index()
        lg = d.groupby(['season', 'position'])['pts_mafia'].transform('mean')
        d['ratio'] = d['pts_mafia'] / lg.replace(0, np.nan)
        d['ratio'] = d['ratio'].fillna(1.0)
        d = d.sort_values(['opp', 'position', 't'])
        dg = d.groupby(['opp', 'position', 'season'])['ratio']
        d['def_n'] = dg.cumcount() + 1
        d['def_cur'] = dg.cumsum() / d['def_n']
        self.def_state = d[['opp', 'position', 'season', 't', 'def_n', 'def_cur']].rename(columns={'season': 'def_season'})
        prev = d.groupby(['opp', 'position', 'season'])['ratio'].mean().reset_index()
        prev['season'] += 1
        self.def_prev = prev.rename(columns={'ratio': 'def_prev'})

    def build(self, q):
        """q: DataFrame with player_id, position, season, week, team, opp (one row per game to predict)."""
        q = q.copy()
        q['season'] = q['season'].astype('int64')
        q['week'] = q['week'].astype('int64')
        q['t'] = q['season'] * 100 + q['week']
        q['_row'] = np.arange(len(q))
        q = q.sort_values('t')

        ps = self.player_state.sort_values('t')
        x = pd.merge_asof(q, ps, on='t', by='player_id', allow_exact_matches=False, direction='backward')
        same = (x['st_season'] == x['season'])
        for c in TRACK:
            x.loc[~same, f'std_{c}'] = 0.0
        x['n_season'] = np.where(same, x['n_season'], 0.0)
        x['has_history'] = x['n_career'].notna().astype(float)
        x['log_career'] = np.log1p(x['n_career'].fillna(0.0))
        gap = np.where(same, x['week'] - x['st_week'], 18 - x['st_week'] + x['week'])
        x['weeks_since_last'] = np.clip(np.nan_to_num(gap, nan=20.0), 0, 20) / 20.0
        x['changed_team'] = ((x['last_team'].notna()) & (x['last_team'] != x['team'])).astype(float)
        x['n_season'] = x['n_season'] / 17.0

        x = x.merge(self.prev_season, on=['player_id', 'season'], how='left')
        x['has_prev'] = x[f'prev_{TRACK[0]}'].notna().astype(float)

        # Opponent: points allowed to this position relative to league average, shrunk to last season
        x = x.sort_values('t')
        ds = self.def_state.sort_values('t')
        x = pd.merge_asof(x, ds, on='t', by=['opp', 'position'], allow_exact_matches=False, direction='backward')
        x = x.merge(self.def_prev, on=['opp', 'position', 'season'], how='left')
        n = np.where(x['def_season'] == x['season'], x['def_n'], 0.0)
        n = np.nan_to_num(n)
        cur = np.nan_to_num(x['def_cur'].values, nan=1.0)
        prev = np.nan_to_num(x['def_prev'].values, nan=1.0)
        x['opp_ratio'] = (n * cur + DEF_PRIOR_GAMES * prev) / (n + DEF_PRIOR_GAMES)

        x = x.merge(self.vegas[['season', 'week', 'team', 'is_home', 'implied_total', 'team_spread']],
                    on=['season', 'week', 'team'], how='left')
        x['is_home'] = x['is_home'].fillna(0.5)
        x['implied_total'] = x['implied_total'].fillna(22.0) / 30.0
        x['team_spread'] = x['team_spread'].fillna(0.0) / 10.0
        x['week_frac'] = x['week'] / 18.0
        for p in POSITIONS:
            x[f'pos_{p}'] = (x['position'] == p).astype(float)

        x = x.sort_values('_row').reset_index(drop=True)
        feats = self.feature_columns()
        x[feats] = x[feats].astype(float).fillna(0.0)
        return x

    @staticmethod
    def feature_columns():
        cols = []
        for pre in ['ewm', 'r8', 'std', 'prev']:
            cols += [f'{pre}_{c}' for c in TRACK]
        cols += ['n_season', 'log_career', 'has_history', 'has_prev', 'weeks_since_last', 'changed_team',
                 'opp_ratio', 'is_home', 'implied_total', 'team_spread', 'week_frac']
        cols += [f'pos_{p}' for p in POSITIONS]
        return cols


# ----------------------------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------------------------
class TwoBranchStatNet(nn.Module):
    def __init__(self, input_dim, stat_mean, stat_std, hidden=(192, 96), dropout=0.15):
        super().__init__()
        layers, prev = [], input_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.LayerNorm(h), nn.GELU(), nn.Dropout(dropout)]
            prev = h
        self.trunk = nn.Sequential(*layers)
        self.stat_head = nn.Linear(prev, len(STAT_COMPONENTS))
        self.sigma_heads = nn.ModuleDict({lg: nn.Linear(prev, 1) for lg in LEAGUES})
        self.register_buffer('stat_mean', torch.tensor(stat_mean, dtype=torch.float32))
        self.register_buffer('stat_std', torch.tensor(stat_std, dtype=torch.float32))
        for lg in LEAGUES:
            self.register_buffer(f'w_{lg}', torch.tensor(scoring_vector(lg)))

    def forward(self, x):
        h = self.trunk(x)
        z = self.stat_head(h)                              # standardized stat line
        stats = z * self.stat_std + self.stat_mean          # expected box score
        out = {'z': z, 'stats': stats}
        for lg in LEAGUES:
            out[f'mu_{lg}'] = stats @ getattr(self, f'w_{lg}')
            out[f'sigma_{lg}'] = nn.functional.softplus(self.sigma_heads[lg](h).squeeze(-1)) + 1.0
        return out


def _loss(out, y_z, y_pts):
    loss = nn.functional.mse_loss(out['z'], y_z)
    for i, lg in enumerate(LEAGUES):
        mu, sig, y = out[f'mu_{lg}'], out[f'sigma_{lg}'], y_pts[:, i]
        loss = loss + nn.functional.mse_loss(mu, y) / 50.0
        # Uncertainty branch learns spread around a frozen mean
        loss = loss + 0.2 * nn.functional.gaussian_nll_loss(mu.detach(), y, sig ** 2)
    return loss


def _fit(model, X, Yz, Yp, epochs, Xv=None, Yzv=None, Ypv=None, lr=2e-3, seed=0):
    torch.manual_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs))
    X, Yz, Yp = map(lambda a: torch.tensor(a, dtype=torch.float32), (X, Yz, Yp))
    if Xv is not None:
        Xv, Yzv, Ypv = map(lambda a: torch.tensor(a, dtype=torch.float32), (Xv, Yzv, Ypv))
    best, best_state, best_ep = float('inf'), None, epochs
    n = len(X)
    for ep in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, 512):
            idx = perm[i:i + 512]
            opt.zero_grad()
            l = _loss(model(X[idx]), Yz[idx], Yp[idx])
            l.backward()
            opt.step()
        sched.step()
        if Xv is not None:
            model.eval()
            with torch.no_grad():
                out = model(Xv)
                vl = _loss(out, Yzv, Ypv).item()
                mae = (out['mu_mafia'] - Ypv[:, 0]).abs().mean().item()
            if vl < best:
                best, best_ep = vl, ep
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
            if ep % 5 == 0 or ep == 1:
                print(f'  epoch {ep:3d}  val loss {vl:.4f}  val MAE (mafia pts) {mae:.3f}')
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_ep


class WeeklyModel:
    def __init__(self):
        self.games = load_games_log()
        self.vegas = load_vegas()
        self.fb = FeatureBuilder(self.games, self.vegas)
        self.feats = FeatureBuilder.feature_columns()
        self.model = None

    def training_frame(self, through_season, through_week=None):
        g = self.games
        mask = g['season'] <= through_season
        if through_week is not None:
            mask &= ~((g['season'] == through_season) & (g['week'] > through_week))
        q = g[mask & (g['season'] >= 2007)][['player_id', 'position', 'season', 'week', 'team', 'opp'] + STAT_COMPONENTS]
        x = self.fb.build(q[['player_id', 'position', 'season', 'week', 'team', 'opp']])
        x[STAT_COMPONENTS] = q[STAT_COMPONENTS].values
        return x

    def _arrays(self, x):
        X = (x[self.feats].values - self.x_mean) / self.x_std
        Y = x[STAT_COMPONENTS].values.astype(np.float32)
        Yz = (Y - self.y_mean) / self.y_std
        Yp = np.stack([Y @ scoring_vector(lg) for lg in LEAGUES], axis=1)
        return X.astype(np.float32), Yz.astype(np.float32), Yp.astype(np.float32)

    def train(self, through_season=2025, through_week=None, epochs=None):
        x = self.training_frame(through_season, through_week)
        print(f'Training rows: {len(x):,} player-games ({x.season.min()}-{through_season}'
              f'{"" if through_week is None else f" wk{through_week}"})')
        self.x_mean = x[self.feats].values.mean(0)
        self.x_std = x[self.feats].values.std(0) + 1e-6
        Y = x[STAT_COMPONENTS].values
        self.y_mean, self.y_std = Y.mean(0), Y.std(0) + 1e-6

        if epochs is None:
            # Pick the epoch count by early stopping on the last full season, then refit on everything
            val_season = 2025
            tr, va = x[x.season < val_season], x[x.season == val_season]
            m = TwoBranchStatNet(len(self.feats), self.y_mean, self.y_std)
            print(f'Selecting epochs (train <{val_season}, validate {val_season})...')
            _, epochs = _fit(m, *self._arrays(tr), epochs=40, Xv=None if va.empty else self._arrays(va)[0],
                             Yzv=None if va.empty else self._arrays(va)[1], Ypv=None if va.empty else self._arrays(va)[2])
            print(f'Best epoch: {epochs}')
        self.model = TwoBranchStatNet(len(self.feats), self.y_mean, self.y_std)
        print(f'Refitting on all rows for {epochs} epochs...')
        self.model, _ = _fit(self.model, *self._arrays(x), epochs=epochs)
        self.model.eval()
        self.trained_through = (through_season, through_week)
        return self

    def save(self, tag='base'):
        os.makedirs(MODEL_DIR, exist_ok=True)
        torch.save(self.model.state_dict(), os.path.join(MODEL_DIR, f'model_{tag}.pt'))
        with open(os.path.join(MODEL_DIR, f'meta_{tag}.pkl'), 'wb') as f:
            pickle.dump({'feats': self.feats, 'x_mean': self.x_mean, 'x_std': self.x_std,
                         'y_mean': self.y_mean, 'y_std': self.y_std, 'trained_through': self.trained_through}, f)
        print(f'Saved weekly model [{tag}] to {MODEL_DIR}')

    def load(self, tag='base'):
        with open(os.path.join(MODEL_DIR, f'meta_{tag}.pkl'), 'rb') as f:
            meta = pickle.load(f)
        self.__dict__.update({k: v for k, v in meta.items()})
        self.model = TwoBranchStatNet(len(self.feats), self.y_mean, self.y_std)
        self.model.load_state_dict(torch.load(os.path.join(MODEL_DIR, f'model_{tag}.pt')))
        self.model.eval()
        return self

    # ------------------------------------------------------------------------------------
    def resolve_players(self, players, season):
        """Map Yahoo (name, pos, team) rows to nflverse player_ids by normalized name (+ position)."""
        g = self.games[self.games['season'] >= season - 3]
        latest = g.sort_values('t').groupby('player_id').tail(1)
        idx = {}
        for r in latest.itertuples():
            idx.setdefault(normalize_name(r.name), []).append((r.player_id, r.position, r.t))
        ids = []
        for r in players.itertuples():
            cands = idx.get(normalize_name(r.player_name), [])
            if len(cands) > 1:
                same_pos = [c for c in cands if c[1] == r.pos]
                cands = same_pos or cands
            if not cands:
                parts = normalize_name(r.player_name).split()
                if len(parts) >= 2:
                    for k, v in idx.items():
                        kp = k.split()
                        if len(kp) >= 2 and kp[-1] == parts[-1] and kp[0][0] == parts[0][0]:
                            cands += [c for c in v if c[1] == r.pos]
            ids.append(max(cands, key=lambda c: c[2])[0] if cands else None)
        return ids

    def predict_players(self, players, season, week):
        """
        players: DataFrame with player_name, pos, team (Yahoo abbreviations ok).
        Returns players with nn2_<league> mean and nn2_sigma_<league> columns (NaN if no game / not skill pos).
        """
        p = players.copy().reset_index(drop=True)
        p['team_std'] = p['team'].map(standardize_team)
        p['player_id'] = self.resolve_players(p, season)
        sched = self.vegas[(self.vegas.season == season) & (self.vegas.week == week)][['team', 'sched_opp']]
        p = p.merge(sched, left_on='team_std', right_on='team', how='left', suffixes=('', '_s'))
        # Unmatched names (no nflverse id) get no projection, so callers can fall back to Yahoo
        ok = p['pos'].isin(POSITIONS) & p['sched_opp'].notna() & p['player_id'].notna()
        q = pd.DataFrame({'player_id': p.loc[ok, 'player_id'].values,
                          'position': p.loc[ok, 'pos'].values, 'season': season, 'week': week,
                          'team': p.loc[ok, 'team_std'].values, 'opp': p.loc[ok, 'sched_opp'].values})
        for lg in LEAGUES:
            p[f'nn2_{lg}'] = np.nan
            p[f'nn2_sigma_{lg}'] = np.nan
        if len(q):
            x = self.fb.build(q)
            X = ((x[self.feats].values - self.x_mean) / self.x_std).astype(np.float32)
            with torch.no_grad():
                out = self.model(torch.tensor(X))
            for lg in LEAGUES:
                p.loc[ok, f'nn2_{lg}'] = out[f'mu_{lg}'].numpy().clip(min=0.0)
                p.loc[ok, f'nn2_sigma_{lg}'] = out[f'sigma_{lg}'].numpy()
        p['nn2_matched'] = p['player_id'].notna()
        return p.drop(columns=['team_s'], errors='ignore')


def yahoo_players(week, league):
    path = os.path.join(CACHE_DIR, f'yahoo_week{week}_all_matchups_{league}.json')
    if not os.path.exists(path):
        return None
    rows, seen = [], set()
    for m in json.load(open(path)):
        for side in ['team1', 'team2']:
            for pl in m[side]['starters'] + m[side]['bench']:
                n = pl.get('player_name')
                if n and n not in seen:
                    seen.add(n)
                    rows.append({'player_name': n, 'pos': pl.get('pos', 'UNK'), 'team': pl.get('team', 'UNK')})
    return pd.DataFrame(rows)


def weekly_preds_path(league, week):
    return os.path.join(CACHE_DIR, f'nn_weekly_preds_{league}_w{week}.parquet')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='Weekly two-branch stat-line NN')
    ap.add_argument('--train', action='store_true')
    ap.add_argument('--through', nargs=2, type=int, metavar=('SEASON', 'WEEK'), help='Include games through this week')
    ap.add_argument('--tag', default=None, help='Model tag (default: base, or live when --through is set)')
    ap.add_argument('--predict', nargs=2, type=int, metavar=('SEASON', 'WEEK'))
    args = ap.parse_args()
    tag = args.tag or ('live' if args.through else 'base')

    wm = WeeklyModel()
    if args.train:
        if args.through:
            wm.train(through_season=args.through[0], through_week=args.through[1])
        else:
            wm.train(through_season=2025)
        wm.save(tag)
    else:
        wm.load(tag)

    if args.predict:
        season, week = args.predict
        for lg in LEAGUES:
            players = yahoo_players(week, lg)
            if players is None:
                print(f'[{lg}] no Yahoo player list for week {week}; run daily_yahoo_sync first')
                continue
            out = wm.predict_players(players, season, week)
            out.to_parquet(weekly_preds_path(lg, week), index=False)
            print(f'[{lg}] week {week}: {out[f"nn2_{lg}"].notna().sum()} projections -> {weekly_preds_path(lg, week)}')
