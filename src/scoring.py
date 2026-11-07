"""
League scoring rules, recovered exactly from Yahoo actual points.

Each league's weights were fit by least squares of Yahoo `actual_pts` against nflverse
2026 weekly box scores (weeks 1-8, ~1,000 player-weeks per league). Both fits reproduce
Yahoo to within 0.01 pt on average, so these are the leagues' real settings, not guesses.

    mafia (10-team, half PPR):   0.04/pass yd, 4/pass TD, -1/INT, 0.5/rec
    12_team_full_ppr:            0.05/pass yd, 6/pass TD, -2/INT, 0.5/completion,
                                 1/rec, 1 pt per 15 return yds

STAT_COMPONENTS is also the output vector of the weekly stat-line network
(src/weekly_model.py); a league's points are the dot product of that vector with
SCORING[league].
"""
import numpy as np
import pandas as pd

STAT_COMPONENTS = [
    'completions', 'passing_yards', 'passing_tds', 'interceptions',
    'rushing_yards', 'rushing_tds',
    'receptions', 'receiving_yards', 'receiving_tds',
    'fumbles_lost', 'two_pt', 'return_yards',
]

SCORING = {
    'mafia': {
        'completions': 0.0, 'passing_yards': 0.04, 'passing_tds': 4.0, 'interceptions': -1.0,
        'rushing_yards': 0.1, 'rushing_tds': 6.0,
        'receptions': 0.5, 'receiving_yards': 0.1, 'receiving_tds': 6.0,
        'fumbles_lost': -2.0, 'two_pt': 2.0, 'return_yards': 0.0,
    },
    '12_team_full_ppr': {
        'completions': 0.5, 'passing_yards': 0.05, 'passing_tds': 6.0, 'interceptions': -2.0,
        'rushing_yards': 0.1, 'rushing_tds': 6.0,
        'receptions': 1.0, 'receiving_yards': 0.1, 'receiving_tds': 6.0,
        'fumbles_lost': -2.0, 'two_pt': 2.0, 'return_yards': 1.0 / 15.0,
    },
}
# The legacy 'default' config was a half-PPR league with the same rules as mafia.
SCORING['default'] = SCORING['mafia']

LEAGUES = ['mafia', '12_team_full_ppr']


def scoring_vector(league):
    rules = SCORING.get(league, SCORING['default'])
    return np.array([rules[c] for c in STAT_COMPONENTS], dtype=np.float32)


def harmonize_stats(df):
    """Map nflverse weekly stats (either the pre-2026 or the 2026 schema) onto STAT_COMPONENTS."""
    df = df.copy()
    renames = {'passing_interceptions': 'interceptions', 'team': 'recent_team',
               'sacks_suffered': 'sacks', 'sack_yards_lost': 'sack_yards'}
    for old, new in renames.items():
        if old in df.columns and new not in df.columns:
            df = df.rename(columns={old: new})

    def col(name):
        return df[name].fillna(0.0) if name in df.columns else pd.Series(0.0, index=df.index)

    df['fumbles_lost'] = col('rushing_fumbles_lost') + col('receiving_fumbles_lost') + col('sack_fumbles_lost')
    df['two_pt'] = col('passing_2pt_conversions') + col('rushing_2pt_conversions') + col('receiving_2pt_conversions')
    df['return_yards'] = col('punt_return_yards') + col('kickoff_return_yards')
    for c in STAT_COMPONENTS:
        df[c] = col(c).astype(float)
    return df


def league_points(df, league):
    """Fantasy points for each row of a harmonized stats frame under a league's rules."""
    rules = SCORING.get(league, SCORING['default'])
    return sum(df[c] * w for c, w in rules.items())

