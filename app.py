import os
import json
import pickle
import torch
import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request, session, redirect, url_for, flash
from numpy.random import normal as npnormal

from src.preprocessing import load_aggregate_player_stats, compute_oline_scores, load_roster_data, prepare_ml_dataset, split_and_scale_data
from src.model import FantasyDataset, FantasyNN, train_model
from src.draft_generator import generate_predict_features, make_draft_list
from src.drive_uploader import upload_file_to_drive

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'fantasy_secret_2026')

# Constants
MODEL_ADP_PATH = "data/cache/fantasy_nn_adp_model.pth"
MODEL_PTS_PATH = "data/cache/fantasy_nn_pts_model.pth"
SCALER_PATH = "data/cache/scaler.pkl"
FEATURES_PATH = "data/cache/feature_cols.pkl"

# 10 League Teams
LEAGUE_TEAMS = [
    'Jonathan Taylor Day', 'Hope Mahomes Still Standing', 'Lizard lizard lizard',
    'MR. SNIFFLES', 'Why did I trade JSN?', 'No Mo Toe Joe',
    'In My Football Era', 'Kittle Me This', 'Rippin Darts', 'Erica Loves Sports'
]

# Global cache for loaded model elements
_model_adp = None
_model_pts = None
_scaler = None
_feature_cols = None

def get_or_train_model():
    global _model_adp, _model_pts, _scaler, _feature_cols
    if _model_adp is not None and _model_pts is not None:
        return _model_adp, _model_pts, _scaler, _feature_cols
        
    if os.path.exists(MODEL_ADP_PATH) and os.path.exists(MODEL_PTS_PATH) and os.path.exists(SCALER_PATH) and os.path.exists(FEATURES_PATH):
        print("Loading saved models and scaler...")
        with open(SCALER_PATH, "rb") as f:
            _scaler = pickle.load(f)
        with open(FEATURES_PATH, "rb") as f:
            _feature_cols = pickle.load(f)
            
        _model_adp = FantasyNN(input_dim=len(_feature_cols))
        _model_adp.load_state_dict(torch.load(MODEL_ADP_PATH))
        _model_adp.eval()
        
        _model_pts = FantasyNN(input_dim=len(_feature_cols))
        _model_pts.load_state_dict(torch.load(MODEL_PTS_PATH))
        _model_pts.eval()
        
        return _model_adp, _model_pts, _scaler, _feature_cols
        
    print("No saved models found. Training new models...")
    df = prepare_ml_dataset()
    X_train, y_train_adp, y_train_pts, X_val, y_val_adp, y_val_pts, scaler, feature_cols = split_and_scale_data(df)
    
    from torch.utils.data import DataLoader
    
    # Train ADP Model
    print("Training ADP Model...")
    train_dataset_adp = FantasyDataset(X_train, y_train_adp)
    val_dataset_adp = FantasyDataset(X_val, y_val_adp)
    train_loader_adp = DataLoader(train_dataset_adp, batch_size=32, shuffle=True)
    val_loader_adp = DataLoader(val_dataset_adp, batch_size=64, shuffle=False)
    
    model_adp = FantasyNN(input_dim=X_train.shape[1], hidden_dims=[128, 64, 32], dropout_prob=0.15)
    model_adp, _, _ = train_model(model_adp, train_loader_adp, val_loader_adp, epochs=60, lr=0.002, patience=10)
    
    # Train Points Model
    print("Training Points Model...")
    train_dataset_pts = FantasyDataset(X_train, y_train_pts)
    val_dataset_pts = FantasyDataset(X_val, y_val_pts)
    train_loader_pts = DataLoader(train_dataset_pts, batch_size=32, shuffle=True)
    val_loader_pts = DataLoader(val_dataset_pts, batch_size=64, shuffle=False)
    
    model_pts = FantasyNN(input_dim=X_train.shape[1], hidden_dims=[128, 64, 32], dropout_prob=0.15)
    model_pts, _, _ = train_model(model_pts, train_loader_pts, val_loader_pts, epochs=60, lr=0.002, patience=10)
    
    os.makedirs(os.path.dirname(MODEL_ADP_PATH), exist_ok=True)
    torch.save(model_adp.state_dict(), MODEL_ADP_PATH)
    torch.save(model_pts.state_dict(), MODEL_PTS_PATH)
    with open(SCALER_PATH, "wb") as f:
        pickle.dump(scaler, f)
    with open(FEATURES_PATH, "wb") as f:
        pickle.dump(feature_cols, f)
        
    _model_adp = model_adp
    _model_pts = model_pts
    _scaler = scaler
    _feature_cols = feature_cols
    _model_adp.eval()
    _model_pts.eval()
    return _model_adp, _model_pts, _scaler, _feature_cols

def generate_predictions(oline_adjustments=None):
    """Loads 2025 stats and generates 2026 predictions with O-line adjustments."""
    model_adp, model_pts, scaler, feature_cols = get_or_train_model()
    
    agg_stats = load_aggregate_player_stats()
    oline_map = compute_oline_scores(agg_stats)
    roster_map = load_roster_data()
    
    # Apply manual O-line adjustments to oline_map
    if oline_adjustments:
        for team, adj in oline_adjustments.items():
            for key in oline_map.keys():
                # oline_map keys are (season, team_abbreviation)
                if key[1] == team:
                    # Update active O-line score
                    oline_map[key] = {
                        'oline_score': oline_map[key]['oline_score'] + float(adj),
                        'team_ypc_ex_qb': oline_map[key]['team_ypc_ex_qb']
                    }
                    
    predict_df = generate_predict_features(agg_stats, oline_map, roster_map)
    draft_list = make_draft_list(model_adp, model_pts, scaler, feature_cols, predict_df)
    return draft_list

@app.before_request
def require_login():
    # Allow static files and the login route itself
    if request.endpoint in ['login', 'static'] or (request.path and request.path.startswith('/static')):
        return
    
    # Check if authenticated
    if not session.get('authenticated'):
        # If it is an API request, return 401 Unauthorized instead of redirecting
        if request.path and request.path.startswith('/api/'):
            return jsonify({'status': 'error', 'message': 'Unauthorized'}), 401
        return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        password = request.form.get('password')
        correct_password = os.environ.get('SITE_PASSWORD', 'amaballs')
        
        if password == correct_password:
            session['authenticated'] = True
            return redirect(url_for('index'))
        else:
            flash('Incorrect password. Please try again.')
            
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('authenticated', None)
    return redirect(url_for('login'))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/players', methods=['GET'])
def get_players():
    try:
        draft_list = generate_predictions()
        players = draft_list.to_dict(orient='records')
        return jsonify({
            'status': 'success',
            'players': players
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/recalculate', methods=['POST'])
def recalculate():
    try:
        data = request.get_json() or {}
        adjustments = data.get('adjustments', {})
        draft_list = generate_predictions(oline_adjustments=adjustments)
        players = draft_list.to_dict(orient='records')
        return jsonify({
            'status': 'success',
            'players': players
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/simulate', methods=['POST'])
def simulate_league():
    try:
        data = request.get_json() or {}
        rosters = data.get('rosters', {})  # Map: {team_name: [list of player dicts]}
        
        # Calculate weekly mean score projections for each team
        # Each team has 15 roster spots. Empty spots are worth 8.0 points/week.
        team_projections = {}
        for team in LEAGUE_TEAMS:
            team_roster = rosters.get(team, [])
            # Calculate sum of weekly averages of drafted players
            drafted_weekly_sum = sum(float(p.get('prev_fantasy_points', 0)) / 14.0 for p in team_roster)
            # Remaining empty slots get 8.0 points/week
            empty_slots = max(0, 15 - len(team_roster))
            weekly_mean = drafted_weekly_sum + (empty_slots * 8.0)
            # Cap the projection to keep it realistic
            team_projections[team] = max(60.0, min(160.0, weekly_mean))

        # Run Monte Carlo Season Simulation (10,000 runs)
        num_simulations = 10000
        playoff_counts = {t: 0 for t in LEAGUE_TEAMS}
        total_wins = {t: 0 for t in LEAGUE_TEAMS}
        total_points = {t: 0.0 for t in LEAGUE_TEAMS}
        
        for _ in range(num_simulations):
            # Start standings for this simulation run
            current_standings = {t: [0, 0.0] for t in LEAGUE_TEAMS} # [wins, points]
            
            # Simulate 14 regular season weeks
            for week in range(1, 15):
                shuffled_teams = list(LEAGUE_TEAMS)
                np.random.shuffle(shuffled_teams)
                
                # Head-to-head pairings
                for i in range(0, len(shuffled_teams), 2):
                    t1, t2 = shuffled_teams[i], shuffled_teams[i+1]
                    
                    # Draw scores from normal distribution
                    score1 = max(0.0, npnormal(team_projections[t1], 20))
                    score2 = max(0.0, npnormal(team_projections[t2], 20))
                    
                    if score1 > score2:
                        current_standings[t1][0] += 1
                        current_standings[t1][1] += score1
                        current_standings[t2][1] += score2
                    else:
                        current_standings[t2][0] += 1
                        current_standings[t2][1] += score2
                        current_standings[t1][1] += score1
                        
            # Determine top 4 playoff teams
            # Sort by wins, then by points for
            sorted_teams = sorted(
                LEAGUE_TEAMS, 
                key=lambda x: (current_standings[x][0], current_standings[x][1]), 
                reverse=True
            )
            for t in sorted_teams[:4]:
                playoff_counts[t] += 1
                
            # Accumulate overall stats for reporting averages
            for t in LEAGUE_TEAMS:
                total_wins[t] += current_standings[t][0]
                total_points[t] += current_standings[t][1]
                
        # Compile results
        leaderboard = []
        for t in LEAGUE_TEAMS:
            avg_w = total_wins[t] / num_simulations
            avg_pts = total_points[t] / num_simulations
            playoff_pct = (playoff_counts[t] / num_simulations) * 100.0
            leaderboard.append({
                'team': t,
                'projected_weekly_avg': round(team_projections[t], 2),
                'avg_wins': round(avg_w, 2),
                'avg_points': round(avg_pts, 2),
                'playoff_probability': round(playoff_pct, 2)
            })
            
        # Sort leaderboard by playoff probability
        leaderboard = sorted(leaderboard, key=lambda x: x['playoff_probability'], reverse=True)
        
        return jsonify({
            'status': 'success',
            'leaderboard': leaderboard
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/export', methods=['POST'])
def export_draft():
    try:
        data = request.get_json() or {}
        rosters = data.get('rosters', {})  # Map: {team_name: [list of player dicts]}
        folder_id = data.get('folder_id', None)
        
        # Compile rosters into a flat DataFrame
        rows = []
        for team, players in rosters.items():
            for p in players:
                rows.append({
                    'League Team': team,
                    'Player Name': p.get('player_name'),
                    'Position': p.get('position'),
                    'NFL Team': p.get('team_predict'),
                    'Predicted ADP': p.get('predicted_adp'),
                    'Predicted Points': p.get('predicted_pts'),
                    'VORP': p.get('vorp'),
                    '2025 Fantasy Points': p.get('prev_fantasy_points'),
                    'O-Line Score': p.get('oline_score')
                })
                
        if len(rows) == 0:
            return jsonify({
                'status': 'error',
                'message': "Cannot export an empty draft board."
            }), 400
            
        df = pd.DataFrame(rows)
        temp_path = "data/cache/draft_rosters_2026.csv"
        os.makedirs(os.path.dirname(temp_path), exist_ok=True)
        df.to_csv(temp_path, index=False)
        
        # Upload to Google Drive
        upload_res = upload_file_to_drive(temp_path, mime_type="text/csv", folder_id=folder_id)
        
        # Remove temp file
        if os.path.exists(temp_path):
            os.remove(temp_path)
            
        if upload_res['success']:
            return jsonify({
                'status': 'success',
                'file_id': upload_res['file_id'],
                'message': upload_res['message']
            })
        else:
            return jsonify({
                'status': 'error',
                'message': upload_res['message']
            }), 500
            
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5002))
    # Pre-load/train model on startup
    get_or_train_model()
    app.run(host='0.0.0.0', port=port, debug=True)
