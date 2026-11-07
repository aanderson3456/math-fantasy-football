"""
Generate season-long NN point predictions for a specific league config, using that
league's own trained model (see the LEAGUE_CONFIG -> model path mapping in app.py).

Usage:
    LEAGUE_CONFIG=mafia            python src/generate_league_predictions.py
    LEAGUE_CONFIG=12_team_full_ppr python src/generate_league_predictions.py

Writes data/cache/nn_preds_<LEAGUE_CONFIG>.parquet, which benchmark_tracker uses for
league-specific NN vs Yahoo scoring. Does not touch live_12team_predictions.parquet.
"""
import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)
os.chdir(BASE_DIR)

from src.config import LEAGUE_CONFIG, CACHE_DIR


def league_preds_path(league):
    return os.path.join(CACHE_DIR, f'nn_preds_{league}.parquet')


if __name__ == '__main__':
    if LEAGUE_CONFIG not in ('mafia', '12_team_full_ppr'):
        print(f"Set LEAGUE_CONFIG to 'mafia' or '12_team_full_ppr' (got '{LEAGUE_CONFIG}').")
        sys.exit(1)

    import app  # model paths are resolved from LEAGUE_CONFIG at import time
    for p in (app.MODEL_ADP_PATH, app.MODEL_PTS_PATH, app.SCALER_PATH, app.FEATURES_PATH):
        if not os.path.exists(p):
            # Refuse to silently train a fresh model in place of the league's saved one.
            print(f"Missing model artifact {p}; aborting.")
            sys.exit(1)

    df = app.generate_predictions()
    keep = [c for c in ['player_name', 'position', 'team_predict', 'predicted_pts', 'vorp'] if c in df.columns]
    out = league_preds_path(LEAGUE_CONFIG)
    df[keep].to_parquet(out, index=False)
    print(f"[{LEAGUE_CONFIG}] Saved {len(df)} predictions from {app.MODEL_PTS_PATH} -> {out}")
