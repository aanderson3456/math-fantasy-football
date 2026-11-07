"""
Brier-score report: Yahoo vs the season-long NN (old and retrained) vs the weekly two-branch NN,
for each league, over every completed 2026 week with a cached Yahoo matchup file.

Two Brier scores per source:
  * Player boom:  P(player scores >= the league/position 70th-percentile week).
  * Matchup win:  P(team 1 beats team 2) for every head-to-head matchup, from the sum of the
                  managers' actual starters' projections.

Fairness rules (identical for every source):
  * Probabilities come from a Gaussian around the source's projection. Its sigma is the source's
    own residual spread by position, estimated only from earlier weeks (walk-forward). Week 1
    uses a flat default. The weekly NN is also scored with its own learned sigma.
  * Availability: when Yahoo projects 0 (bye, OUT, IR) every NN projection is set to 0 too, and
    those player-weeks are left out of the player metrics. Yahoo's injury tag is not used: on
    back-filled weeks it is the tag on the scrape date, not on game day.
  * Player metrics use only player-weeks that every source projects (QB/RB/WR/TE matched in all).
  * In matchup sums, a starter a source can't project (K, DEF, unmatched) uses Yahoo's number.
  * The weekly NN used here is trained through 2025 only, so 2026 is fully out of sample.

Usage:  python src/brier_report.py [--weeks 1-8]
"""
import os
import sys
import json
import argparse

import numpy as np
import pandas as pd
from scipy.stats import norm, spearmanr

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.config import CACHE_DIR
from src.scoring import LEAGUES
from src.weekly_model import WeeklyModel, normalize_name, weekly_preds_path, POSITIONS

BENCHMARK_DIR = os.path.join(BASE_DIR, 'data', 'benchmark')
OLD_PREDS_DIR = os.path.join(BASE_DIR, 'data', 'models_backup_20261106')
OUT_STATUSES = {'O', 'IR', 'PUP', 'SUSP', 'NA', 'IR-R', 'NFI', 'COV'}
DEFAULT_SIGMA = 7.0
MIN_RESID = 25
BOOM_PCTL = 70

SOURCES = ['yahoo', 'nn_season_old', 'nn_season_new', 'nn_weekly', 'blend']
LABELS = {'yahoo': 'Yahoo', 'nn_season_old': 'NN season (old)', 'nn_season_new': 'NN season (retrained)',
          'nn_weekly': 'NN weekly (two-branch)', 'nn_weekly_learned': 'NN weekly, learned σ',
          'blend': 'NN weekly + Yahoo blend'}
WF_EPOCHS = 34  # epoch count chosen by early stopping when the base model was trained


def season_lookup(path):
    if not os.path.exists(path):
        return {}
    df = pd.read_parquet(path)
    out = {}
    for r in df.itertuples():
        out.setdefault(normalize_name(r.player_name), []).append((str(r.position), float(r.predicted_pts) / 17.0))
    return out


def lookup(table, name, pos):
    c = table.get(normalize_name(name))
    if not c:
        parts = normalize_name(name).split()
        if len(parts) >= 2:
            c = [v for k, vs in table.items() for v in vs
                 if len(k.split()) >= 2 and k.split()[-1] == parts[-1] and k[0] == parts[0][0]]
    if not c:
        return np.nan
    same = [v for p, v in c if p == pos]
    return same[0] if same else c[0][1]


def load_week(league, week):
    path = os.path.join(CACHE_DIR, f'yahoo_week{week}_all_matchups_{league}.json')
    if not os.path.exists(path):
        return None, None
    matchups = json.load(open(path))
    rows = []
    for mi, m in enumerate(matchups):
        for side in ['team1', 'team2']:
            for slot_group, starter in [('starters', True), ('bench', False)]:
                for pl in m[side][slot_group]:
                    if not pl.get('player_name'):
                        continue
                    rows.append({'week': week, 'matchup': mi, 'side': side, 'starter': starter,
                                 'player_name': pl['player_name'], 'pos': pl.get('pos', 'UNK'),
                                 'team': pl.get('team', 'UNK'), 'status': pl.get('status'),
                                 'yahoo': float(pl.get('yahoo_proj') or 0.0),
                                 'actual': float(pl['actual_pts']) if 'actual_pts' in pl else np.nan})
    return pd.DataFrame(rows), matchups


def walk_forward_sigma(resid_hist, src, pos, week):
    r = resid_hist.get((src, pos), [])
    r = [e for w, e in r if w < week]
    return float(np.sqrt(np.mean(np.square(r)))) if len(r) >= MIN_RESID else DEFAULT_SIGMA


def brier(p, o):
    p, o = np.asarray(p, float), np.asarray(o, float)
    bs = np.mean((p - o) ** 2)
    base = o.mean()
    ref = base * (1 - base)
    return bs, (1 - bs / ref) if ref > 0 else np.nan


def weekly_model_for(week, walk_forward, cache={}):
    """Base model (trained through 2025), or with walk_forward a model also fit on 2026 weeks < week."""
    tag = 'base' if (not walk_forward or week <= 1) else f'wf_2026_w{week - 1}'
    if tag not in cache:
        wm = cache.get('_shared') or WeeklyModel()
        cache['_shared'] = wm
        m = WeeklyModel.__new__(WeeklyModel)
        m.__dict__.update({k: wm.__dict__[k] for k in ('games', 'vegas', 'fb', 'feats')})
        try:
            m.load(tag)
        except FileNotFoundError:
            print(f'Training weekly model [{tag}]...')
            m.train(through_season=2026, through_week=week - 1, epochs=WF_EPOCHS)
            m.save(tag)
        cache[tag] = m
    return cache[tag]


def run(weeks, walk_forward=False):
    report = {}
    for league in LEAGUES:
        old_tbl = season_lookup(os.path.join(OLD_PREDS_DIR, f'nn_preds_{league}.parquet'))
        new_tbl = season_lookup(os.path.join(CACHE_DIR, f'nn_preds_{league}.parquet'))
        frames, all_matchups = [], {}
        for w in weeks:
            df, matchups = load_week(league, w)
            if df is None:
                continue
            all_matchups[w] = matchups
            uniq = df.drop_duplicates('player_name')[['player_name', 'pos', 'team']]
            pred = weekly_model_for(w, walk_forward).predict_players(uniq, 2026, w)
            pred.to_parquet(weekly_preds_path(league, w), index=False)
            pred = pred.set_index('player_name')
            df['nn_weekly'] = df['player_name'].map(pred[f'nn2_{league}'])
            df['nn_weekly_sigma'] = df['player_name'].map(pred[f'nn2_sigma_{league}'])
            df['nn_season_old'] = [lookup(old_tbl, n, p) if p in POSITIONS else np.nan for n, p in zip(df.player_name, df.pos)]
            df['nn_season_new'] = [lookup(new_tbl, n, p) if p in POSITIONS else np.nan for n, p in zip(df.player_name, df.pos)]
            # Yahoo's 'status' on back-filled weeks is the player's status on the scrape date, not
            # game day (Hurts shows IR on his 39.75-pt week 1), so only a 0 projection means "out".
            out = df['yahoo'] <= 0
            for s in ['nn_weekly', 'nn_season_old', 'nn_season_new']:
                df.loc[out & df[s].notna(), s] = 0.0
            frames.append(df)
        if not frames:
            continue
        d = pd.concat(frames, ignore_index=True)
        d['actual'] = d['actual']

        # ---- player metrics on the common set
        ev = d[d.pos.isin(POSITIONS) & d.actual.notna() & (d.yahoo > 0)].drop_duplicates(['week', 'player_name'])
        ev = ev.dropna(subset=['nn_weekly', 'nn_season_old', 'nn_season_new'])

        # NN + Yahoo blend: weight on the NN fit by least squares on earlier weeks only
        alphas = {}
        for w in sorted(d.week.unique()):
            past = ev[ev.week < w]
            if len(past) < 50:
                alphas[w] = 0.5
            else:
                diff = (past['nn_weekly'] - past['yahoo']).values
                alphas[w] = float(np.clip(np.dot(diff, (past['actual'] - past['yahoo']).values) / max(np.dot(diff, diff), 1e-9), 0, 1))
        a = d['week'].map(alphas)
        d['blend'] = np.where(d['nn_weekly'].notna(), a * d['nn_weekly'] + (1 - a) * d['yahoo'], np.nan)
        ev['blend'] = ev['week'].map(alphas) * ev['nn_weekly'] + (1 - ev['week'].map(alphas)) * ev['yahoo']
        thresholds = ev.groupby('pos')['actual'].quantile(BOOM_PCTL / 100).to_dict()
        ev['boom'] = (ev['actual'] >= ev['pos'].map(thresholds)).astype(float)

        resid_hist = {}
        probs = {s: [] for s in SOURCES + ['nn_weekly_learned']}
        for w in sorted(ev.week.unique()):
            e = ev[ev.week == w]
            for s in SOURCES:
                sig = np.array([walk_forward_sigma(resid_hist, s, p, w) for p in e.pos])
                probs[s].append(pd.Series(1 - norm.cdf(e.pos.map(thresholds).values, loc=e[s].values, scale=sig), index=e.index))
            probs['nn_weekly_learned'].append(pd.Series(
                1 - norm.cdf(e.pos.map(thresholds).values, loc=e['nn_weekly'].values, scale=e['nn_weekly_sigma'].values), index=e.index))
            for s in SOURCES:
                for p, r in zip(e.pos, e[s] - e.actual):
                    resid_hist.setdefault((s, p), []).append((w, r))
        for s in probs:
            ev[f'p_{s}'] = pd.concat(probs[s])

        player = {}
        for s in SOURCES + ['nn_weekly_learned']:
            col = s if s != 'nn_weekly_learned' else 'nn_weekly'
            bs, bss = brier(ev[f'p_{s}'], ev['boom'])
            rho = np.mean([spearmanr(g[col], g.actual)[0] for _, g in ev.groupby('week')])
            player[s] = {'mae': float((ev[col] - ev.actual).abs().mean()),
                         'bias': float((ev[col] - ev.actual).mean()),
                         'rank_corr': float(rho), 'boom_brier': float(bs), 'boom_bss': float(bss)}
        by_pos = {}
        for pos, g in ev.groupby('pos'):
            by_pos[pos] = {s: {'mae': float((g[s] - g.actual).abs().mean()),
                               'bias': float((g[s] - g.actual).mean()),
                               'boom_brier': float(np.mean((g[f'p_{s}'] - g.boom) ** 2))} for s in SOURCES}
        by_week = {}
        for w, g in ev.groupby('week'):
            by_week[int(w)] = {s: {'mae': float((g[s] - g.actual).abs().mean()),
                                   'boom_brier': float(np.mean((g[f'p_{s}'] - g.boom) ** 2))} for s in SOURCES}

        # ---- matchup win probabilities
        mrows = []
        for w, matchups in all_matchups.items():
            dw = d[(d.week == w) & d.starter]
            for mi, m in enumerate(matchups):
                a1, a2 = m['team1'].get('actual_total'), m['team2'].get('actual_total')
                if a1 is None or a2 is None or a1 == a2:
                    continue
                rec = {'week': w, 'matchup': mi, 'team1_won': float(a1 > a2)}
                for s in SOURCES + ['nn_weekly_learned']:
                    col = s if s != 'nn_weekly_learned' else 'nn_weekly'
                    tot, var = {}, {}
                    for side in ['team1', 'team2']:
                        st = dw[(dw.matchup == mi) & (dw.side == side)]
                        proj = st[col].where(st[col].notna(), st['yahoo'])
                        if s == 'nn_weekly_learned':
                            sig = np.where(st['nn_weekly'].notna(), st['nn_weekly_sigma'].fillna(DEFAULT_SIGMA), DEFAULT_SIGMA)
                        else:
                            src = np.where(st[col].notna(), s, 'yahoo')
                            sig = np.array([walk_forward_sigma(resid_hist, ss, p, w) for ss, p in zip(src, st.pos)])
                        sig = np.where(st['yahoo'] <= 0, 0.0, sig)
                        tot[side], var[side] = proj.sum(), np.sum(sig ** 2)
                    rec[f'p_{s}'] = float(norm.cdf((tot['team1'] - tot['team2']) / np.sqrt(var['team1'] + var['team2'] + 1e-9)))
                mrows.append(rec)
        md = pd.DataFrame(mrows)
        matchup = {}
        for s in SOURCES + ['nn_weekly_learned']:
            bs, bss = brier(md[f'p_{s}'], md['team1_won'])
            matchup[s] = {'brier': float(bs), 'bss': float(bss),
                          'picked_winner_pct': float(((md[f'p_{s}'] > 0.5) == (md.team1_won == 1)).mean() * 100)}

        report[league] = {'weeks': sorted(int(w) for w in ev.week.unique()), 'player_weeks': int(len(ev)),
                          'matchups': int(len(md)), 'boom_thresholds': {k: round(v, 2) for k, v in thresholds.items()},
                          'blend_nn_weight_by_week': {int(k): round(v, 3) for k, v in alphas.items()},
                          'player': player, 'by_position': by_pos, 'by_week': by_week, 'matchup': matchup}
    return report


def print_report(rep):
    for league, r in rep.items():
        print(f"\n=== {league}: weeks {r['weeks'][0]}-{r['weeks'][-1]}, {r['player_weeks']} player-weeks, {r['matchups']} matchups ===")
        print(f"Boom thresholds (70th pct): {r['boom_thresholds']}")
        print(f"Blend NN weight by week: {r['blend_nn_weight_by_week']}")
        print(f"{'source':28s} {'MAE':>6s} {'bias':>6s} {'rank ρ':>7s} {'boom BS':>8s} {'BSS':>6s} | {'match BS':>8s} {'BSS':>6s} {'picked':>7s}")
        for s in SOURCES + ['nn_weekly_learned']:
            p, m = r['player'][s], r['matchup'][s]
            print(f"{LABELS[s]:28s} {p['mae']:6.2f} {p['bias']:+6.2f} {p['rank_corr']:7.3f} {p['boom_brier']:8.4f} {p['boom_bss']:+6.3f} | "
                  f"{m['brier']:8.4f} {m['bss']:+6.3f} {m['picked_winner_pct']:6.1f}%")
        print('By position (MAE / bias):')
        for pos, v in r['by_position'].items():
            print(f"  {pos}: " + '  '.join(f"{LABELS[s].split(' (')[0] if s == 'yahoo' else s}: {v[s]['mae']:.2f}/{v[s]['bias']:+.2f}" for s in SOURCES))
        print('By week (MAE yahoo / nn_weekly):', {w: f"{v['yahoo']['mae']:.2f}/{v['nn_weekly']['mae']:.2f}" for w, v in r['by_week'].items()})


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--weeks', default='1-8')
    ap.add_argument('--walk-forward', action='store_true',
                    help='Score week w with a weekly model also trained on 2026 weeks < w')
    args = ap.parse_args()
    a, b = args.weeks.split('-')
    rep = run(range(int(a), int(b) + 1), walk_forward=args.walk_forward)
    os.makedirs(BENCHMARK_DIR, exist_ok=True)
    name = 'brier_report_walkforward.json' if args.walk_forward else 'brier_report.json'
    with open(os.path.join(BENCHMARK_DIR, name), 'w') as f:
        json.dump(rep, f, indent=2)
    print_report(rep)
