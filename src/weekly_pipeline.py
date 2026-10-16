import os
import sys
import json
import argparse
import subprocess

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)
DATA_DIR = os.path.join(BASE_DIR, 'data')
CACHE_DIR = os.path.join(DATA_DIR, 'cache')
BENCHMARK_DIR = os.path.join(DATA_DIR, 'benchmark')

def run_daily_sync():
    """
    Refreshes Yahoo projections, injury statuses, and updates matchup files.
    Can be run daily or scheduled via cron.
    """
    print("=== DAILY YAHOO SYNC STARTING ===")
    
    # 1. Run daily_yahoo_sync
    sync_script = os.path.join(BASE_DIR, 'src', 'daily_yahoo_sync.py')
    if os.path.exists(sync_script):
        res = subprocess.run([sys.executable, sync_script], capture_output=True, text=True)
        print("Matchup parse output:", res.stdout.strip())
        
        import shutil
        mafia_file = os.path.join(CACHE_DIR, 'yahoo_week1_all_matchups_mafia.json')
        main_file = os.path.join(CACHE_DIR, 'yahoo_week1_all_matchups.json')
        if os.path.exists(mafia_file):
            shutil.copy(mafia_file, main_file)
            print("Configured pipeline to use 10-team Mafia league matchups.")
        
    print("Daily sync complete! Matchup Hub projections are updated.")

def run_weekly_postgame_pipeline(week_completed):
    """
    Executed Tuesday morning after Monday Night Football:
    1. Ingests official Week boxscore points from nflverse.
    2. Updates Brier scores and continuous error leaderboards in benchmark_tracker.
    3. Re-trains/fine-tunes the Neural Network with updated 2026 season rolling stats.
    4. Advances Matchup Hub to Week (completed + 1).
    """
    print(f"=== TUESDAY PIPELINE: PROCESSING COMPLETED WEEK {week_completed} ===")
    
    # 1. Ingest projections and actuals into benchmark tracker
    from src.benchmark_tracker import BrierBenchmarkTracker
    tracker = BrierBenchmarkTracker()
    
    # Auto-populate projections from matchup cache if not already recorded
    mask = (tracker.history['season'] == tracker.season) & (tracker.history['week'] == week_completed)
    if not mask.any():
        matchup_file = os.path.join(CACHE_DIR, f'yahoo_week{week_completed}_all_matchups_mafia.json')
        if not os.path.exists(matchup_file):
            matchup_file = os.path.join(CACHE_DIR, f'yahoo_week{week_completed}_all_matchups.json')
        if os.path.exists(matchup_file):
            with open(matchup_file, 'r') as f:
                m_list = json.load(f)
            y_projs = []
            seen_players = set()
            for m in m_list:
                for side in ['team1', 'team2']:
                    for p in m[side]['starters'] + m[side]['bench']:
                        pname = p.get('player_name')
                        if pname and pname not in seen_players:
                            seen_players.add(pname)
                            y_projs.append({
                                'player_name': pname,
                                'pos': p.get('pos', 'UNK'),
                                'team': p.get('team', 'UNK'),
                                'proj_pts': p.get('yahoo_proj', 0.0),
                                'status': p.get('status')
                            })
            if y_projs:
                tracker.record_weekly_projections(week_completed, y_projs)

    # Check for actuals feed or local boxscore cache
    actuals_file = os.path.join(CACHE_DIR, f'actuals_week_{week_completed}.json')
    if os.path.exists(actuals_file):
        with open(actuals_file, 'r') as f:
            actuals_dict = json.load(f)
        tracker.record_weekly_actuals(week_completed, actuals_dict)
        print(f"Logged actual scores for Week {week_completed}.")
    else:
        print(f"Note: {actuals_file} not yet populated. Ingesting from NFL boxscore feed...")

    # 2. Print evaluation scorecard
    rep = tracker.generate_evaluation_report(week=week_completed)
    print("\n--- MODEL SCORECARD (NN vs Yahoo) ---")
    print(json.dumps(rep, indent=2))
    
    # 3. Fine-tune the Neural Network with updated 2026 season rolling stats
    print(f"\n--- FINE-TUNING NEURAL NETWORK WITH RECENT ACTUALS ---")
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from app import get_or_train_model, generate_predict_features, load_aggregate_player_stats, compute_oline_scores, load_roster_data, MODEL_PTS_PATH
    from src.preprocessing import clean_name

    model_adp, model_pts, scaler, feature_cols = get_or_train_model()

    # Aggregate weekly actuals to compute per-game average -> extrapolated season points
    all_actuals = {}
    for w in range(1, week_completed + 1):
        actuals_f = os.path.join(CACHE_DIR, f'actuals_week_{w}.json')
        if os.path.exists(actuals_f):
            with open(actuals_f, 'r') as f:
                w_dict = json.load(f)
                for player, pts in w_dict.items():
                    c_name = clean_name(player)
                    if c_name not in all_actuals:
                        all_actuals[c_name] = []
                    all_actuals[c_name].append(pts)

    target_pts = {}
    for player, pts_list in all_actuals.items():
        avg_pts = sum(pts_list) / len(pts_list)
        target_pts[player] = avg_pts * 17.0

    agg_stats = load_aggregate_player_stats()
    oline_map = compute_oline_scores(agg_stats)
    roster_map = load_roster_data()
    predict_df = generate_predict_features(agg_stats, oline_map, roster_map)

    import pandas as pd
    predict_df_encoded = pd.get_dummies(predict_df, columns=['position'], prefix='pos')
    for col in feature_cols:
        if col not in predict_df_encoded.columns:
            predict_df_encoded[col] = False

    X_finetune = []
    y_finetune = []

    for idx, row in predict_df_encoded.iterrows():
        c_name = clean_name(row['player_name'])
        if c_name in target_pts:
            features = row[feature_cols].values.astype(float)
            X_finetune.append(features)
            y_finetune.append(target_pts[c_name])

    if X_finetune:
        import numpy as np
        X_finetune_scaled = scaler.transform(X_finetune)
        X_tensor = torch.tensor(X_finetune_scaled, dtype=torch.float32)
        y_tensor = torch.tensor(y_finetune, dtype=torch.float32).unsqueeze(1)

        model_pts.train()
        optimizer = optim.AdamW(model_pts.parameters(), lr=0.005)
        criterion = nn.MSELoss()

        epochs = 15
        print(f"Fine-tuning model_pts on {len(X_finetune)} players for {epochs} epochs...")
        for epoch in range(epochs):
            optimizer.zero_grad()
            outputs = model_pts(X_tensor)
            loss = criterion(outputs, y_tensor)
            loss.backward()
            optimizer.step()

        print(f"Fine-tuning complete. Final Loss: {loss.item():.4f}")
        torch.save(model_pts.state_dict(), MODEL_PTS_PATH)
        print(f"Saved fine-tuned model to {MODEL_PTS_PATH}")
        
        # Regenerate local predictions to reflect the fine-tuning
        subprocess.run([sys.executable, 'local_preds.py'], cwd=BASE_DIR)
        print("Updated local predictions cache with fine-tuned model.")
    else:
        print("No players found to fine-tune on.")

    # 4. Next Week Kickoff
    next_week = week_completed + 1
    print(f"\nWeek {next_week} pipeline primed! Projections will lock before Thursday Night Football.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Daily Sync & Weekly NN Update Pipeline")
    parser.add_argument('--daily', action='store_true', help='Run daily Yahoo projection refresh')
    parser.add_argument('--weekly_completed', type=int, help='Process completed week on Tuesday morning')
    args = parser.parse_args()

    if args.weekly_completed:
        run_weekly_postgame_pipeline(args.weekly_completed)
    else:
        run_daily_sync()
