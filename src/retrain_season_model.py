"""
Retrain a league's season-long (preseason) ADP + points models from scratch with the
league's exact scoring (src/scoring.py), then regenerate nn_preds_<league>.parquet.

Usage:
    LEAGUE_CONFIG=12_team_full_ppr python src/retrain_season_model.py
    LEAGUE_CONFIG=mafia            python src/retrain_season_model.py

The previous artifacts are moved to data/models_backup_<timestamp>/ first.
"""
import os
import sys
import shutil
import time

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)
os.chdir(BASE_DIR)

from src.config import LEAGUE_CONFIG, CACHE_DIR

if __name__ == '__main__':
    if LEAGUE_CONFIG not in ('mafia', '12_team_full_ppr'):
        print(f"Set LEAGUE_CONFIG to 'mafia' or '12_team_full_ppr' (got '{LEAGUE_CONFIG}').")
        sys.exit(1)

    import app
    backup = os.path.join(BASE_DIR, 'data', f"models_backup_{time.strftime('%Y%m%d_%H%M%S')}_{LEAGUE_CONFIG}")
    os.makedirs(backup, exist_ok=True)
    for p in (app.MODEL_ADP_PATH, app.MODEL_PTS_PATH, app.SCALER_PATH, app.FEATURES_PATH):
        if os.path.exists(p):
            shutil.move(p, os.path.join(backup, os.path.basename(p)))
    print(f"[{LEAGUE_CONFIG}] Old artifacts moved to {backup}")

    app.get_or_train_model()  # no artifacts on disk -> trains and saves fresh models

    df = app.generate_predictions()
    keep = [c for c in ['player_name', 'position', 'team_predict', 'predicted_pts', 'vorp'] if c in df.columns]
    out = os.path.join(CACHE_DIR, f'nn_preds_{LEAGUE_CONFIG}.parquet')
    df[keep].to_parquet(out, index=False)
    print(f"[{LEAGUE_CONFIG}] Saved {len(df)} predictions -> {out}")
    top = df.sort_values('predicted_pts', ascending=False)
    for pos in ['QB', 'RB', 'WR', 'TE']:
        t = top[top['position'] == pos].head(3)
        print(pos, ', '.join(f"{r.player_name} {r.predicted_pts:.0f}" for r in t.itertuples()))

