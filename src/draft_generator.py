import torch
import pandas as pd
import numpy as np
from src.config import standardize_team, PREDICT_YEAR, TEAM_OLINE_ADJUSTMENTS
from src.preprocessing import clean_name

# Dict of major player team movements for the prediction season
# This allows the model to pair a player with their NEW team's offensive line
PLAYER_TEAM_CHANGES = {
    # 'saquon barkley': 'PHI',
    # 'derrick henry': 'BAL',
    # 'austin ekeler': 'WAS',
}

from src.preprocessing import load_draft_picks

# List of top incoming 2026 rookie prospects (including Notre Dame RBs)
ROOKIE_PROSPECTS_2026 = [
    # Notre Dame 2026 NFL Rookies (Entering Sept 2026)
    {'name': 'Jeremiyah Love', 'pos': 'RB', 'team': 'CHI', 'pick': 24, 'round': 1, 'college': 'Notre Dame'},
    {'name': 'Jeremiah Price', 'pos': 'RB', 'team': 'IND', 'pick': 110, 'round': 4, 'college': 'Notre Dame'},
    {'name': 'Aneyas Williams', 'pos': 'RB', 'team': 'DET', 'pick': 120, 'round': 4, 'college': 'Notre Dame'},
    {'name': 'Kedren Young', 'pos': 'RB', 'team': 'GB', 'pick': 135, 'round': 5, 'college': 'Notre Dame'},
    {'name': 'Mitchell Evans', 'pos': 'TE', 'team': 'GB', 'pick': 85, 'round': 3, 'college': 'Notre Dame'},
    
    # Top 2026 NFL Draft Running Back Rookies
    {'name': 'Ashton Jeanty', 'pos': 'RB', 'team': 'DAL', 'pick': 12, 'round': 1, 'college': 'Boise State'},
    {'name': 'Quinshon Judkins', 'pos': 'RB', 'team': 'HOU', 'pick': 38, 'round': 2, 'college': 'Ohio State'},
    {'name': 'TreVeyon Henderson', 'pos': 'RB', 'team': 'NE', 'pick': 45, 'round': 2, 'college': 'Ohio State'},
    {'name': 'Ollie Gordon II', 'pos': 'RB', 'team': 'MIA', 'pick': 62, 'round': 2, 'college': 'Oklahoma State'},
    {'name': 'Omarion Hampton', 'pos': 'RB', 'team': 'LAC', 'pick': 35, 'round': 2, 'college': 'North Carolina'},
    {'name': 'Nicholas Singleton', 'pos': 'RB', 'team': 'WAS', 'pick': 75, 'round': 3, 'college': 'Penn State'},
    {'name': 'Kaleb Johnson', 'pos': 'RB', 'team': 'GB', 'pick': 50, 'round': 2, 'college': 'Iowa'},
    
    # Top 2026 NFL Draft Wide Receiver Rookies
    {'name': 'Travis Hunter', 'pos': 'WR', 'team': 'JAX', 'pick': 2, 'round': 1, 'college': 'Colorado'},
    {'name': 'Tetairoa McMillan', 'pos': 'WR', 'team': 'CAR', 'pick': 8, 'round': 1, 'college': 'Arizona'},
    {'name': 'Luther Burden III', 'pos': 'WR', 'team': 'NE', 'pick': 15, 'round': 1, 'college': 'Missouri'},
    {'name': 'Emeka Egbuka', 'pos': 'WR', 'team': 'KC', 'pick': 30, 'round': 1, 'college': 'Ohio State'},
    {'name': 'Elic Ayomanor', 'pos': 'WR', 'team': 'SF', 'pick': 40, 'round': 2, 'college': 'Stanford'},
    {'name': 'Isaiah Bond', 'pos': 'WR', 'team': 'BUF', 'pick': 48, 'round': 2, 'college': 'Texas'},

    # Top 2026 NFL Draft Quarterback Rookies
    {'name': 'Cam Ward', 'pos': 'QB', 'team': 'NYG', 'pick': 5, 'round': 1, 'college': 'Miami'},
    {'name': 'Shedeur Sanders', 'pos': 'QB', 'team': 'LV', 'pick': 10, 'round': 1, 'college': 'Colorado'},
    {'name': 'Jaxson Dart', 'pos': 'QB', 'team': 'TEN', 'pick': 22, 'round': 1, 'college': 'Ole Miss'},
    {'name': 'Quinn Ewers', 'pos': 'QB', 'team': 'PIT', 'pick': 33, 'round': 2, 'college': 'Texas'},

    # Top 2026 NFL Draft Tight End Rookies
    {'name': 'Colston Loveland', 'pos': 'TE', 'team': 'IND', 'pick': 25, 'round': 1, 'college': 'Michigan'},
    {'name': 'Tyler Warren', 'pos': 'TE', 'team': 'IND', 'pick': 42, 'round': 2, 'college': 'Penn State'}
]

def generate_predict_features(player_stats_df, oline_map, roster_map):
    """
    Constructs the features for the upcoming draft prediction year using
    stats from PREDICT_YEAR - 1 (the last completed season) as Y-1 features,
    projecting age, experience, draft capital, and mapping players to their teams.
    """
    stats_year = PREDICT_YEAR - 1
    print(f"Preparing features for the {PREDICT_YEAR} season draft (using {stats_year} performance stats)...")
    
    draft_map = load_draft_picks()
    
    # Filter for stats_year stats (which will serve as Y-1 stats for PREDICT_YEAR draft)
    stats_prev = player_stats_df[player_stats_df['season'] == stats_year].copy()
    stats_prev['clean_name'] = stats_prev['player_name'].apply(clean_name)
    stats_prev['std_team'] = stats_prev['recent_team'].apply(standardize_team)
    
    # Use stats_year final O-line scores as the baseline proxy for PREDICT_YEAR lines
    oline_prev_map = {}
    for (season, team), metrics in oline_map.items():
        if season == stats_year:
            oline_prev_map[team] = metrics
            
    # Default O-line stats for missing teams
    default_oline = {'oline_score': 0.0, 'team_ypc_ex_qb': 4.0}
    
    predict_rows = []
    processed_names = set()
    
    # 1. Process Veterans & Existing Players
    for _, row in stats_prev.iterrows():
        name = row['player_name']
        pos = str(row['position']).upper()
        if pos not in ['QB', 'RB', 'WR', 'TE']:
            continue
            
        cname = row['clean_name']
        processed_names.add((cname, pos))
        
        # Determine their team for PREDICT_YEAR (check transfer dictionary first)
        team_predict = PLAYER_TEAM_CHANGES.get(cname, row['std_team'])
        
        # Get their age and experience for PREDICT_YEAR
        roster_key = (stats_year, cname, pos)
        years_exp = 1  # default
        age = 26.0  # default
        
        roster_info = roster_map.get(roster_key, None)
        if roster_info:
            years_exp = roster_info['years_exp'] + 1
            birth_date = roster_info['birth_date']
            if birth_date:
                try:
                    birth_year = pd.to_datetime(birth_date).year
                    age = PREDICT_YEAR - birth_year
                except:
                    pass
        else:
            years_exp = 1
            age = 23.0
            
        # Get O-line metrics of their team
        oline_info = oline_prev_map.get(team_predict, default_oline)
        oline_score = oline_info['oline_score'] + TEAM_OLINE_ADJUSTMENTS.get(team_predict, 0.0)
        team_ypc = oline_info['team_ypc_ex_qb']

        # Pillar 1 Draft Capital
        draft_info = draft_map.get((cname, pos), {'pick': 250, 'round': 8, 'college': 'Other'})
        draft_pick_num = draft_info['pick']
        draft_round = draft_info['round']
        is_day_1 = 1.0 if draft_round == 1 else 0.0
        is_day_2 = 1.0 if draft_round in [2, 3] else 0.0
        is_rookie_top_50 = 0.0
        rookie_expected_volume = 0.0
        
        # Create row
        is_rb = 1.0 if pos == 'RB' else 0.0
        row_dict = {
            'player_name': name,
            'position': pos,
            'team_predict': team_predict,
            'is_rookie': 0,
            'years_exp': years_exp,
            'age': age,
            'oline_score': oline_score,
            'team_ypc_ex_qb': team_ypc,
            # Pillar 1 Features
            'draft_pick_num': draft_pick_num,
            'draft_round': draft_round,
            'is_day_1': is_day_1,
            'is_day_2': is_day_2,
            'is_rookie_top_50': is_rookie_top_50,
            'rookie_expected_volume': rookie_expected_volume,
            # prev season stats
            'prev_passing_yards': row['passing_yards'],
            'prev_passing_tds': row['passing_tds'],
            'prev_interceptions': row['interceptions'],
            'prev_rushing_yards': row['rushing_yards'],
            'prev_carries': row['carries'],
            'prev_rushing_tds': row['rushing_tds'],
            'prev_receptions': row['receptions'],
            'prev_targets': row['targets'],
            'prev_receiving_yards': row['receiving_yards'],
            'prev_receiving_tds': row['receiving_tds'],
            'prev_fumbles_lost': row['fumbles_lost'],
            'prev_fantasy_points': row['fantasy_points'],
            # Interaction terms
            'rb_oline_score_inter': is_rb * oline_score,
            'rb_team_ypc_inter': is_rb * team_ypc,
            'wr_oline_score_inter': (1.0 if pos == 'WR' else 0.0) * oline_score,
            'te_oline_score_inter': (1.0 if pos == 'TE' else 0.0) * oline_score
        }
        predict_rows.append(row_dict)

    # 2. Add Incoming 2026 Rookie Prospects (including Notre Dame RBs)
    for r in ROOKIE_PROSPECTS_2026:
        r_cname = clean_name(r['name'])
        if (r_cname, r['pos']) in processed_names:
            continue
            
        team_predict = r['team']
        pos = r['pos']
        oline_info = oline_prev_map.get(team_predict, default_oline)
        oline_score = oline_info['oline_score'] + TEAM_OLINE_ADJUSTMENTS.get(team_predict, 0.0)
        team_ypc = oline_info['team_ypc_ex_qb']

        draft_pick_num = r['pick']
        draft_round = r['round']
        is_day_1 = 1.0 if draft_round == 1 else 0.0
        is_day_2 = 1.0 if draft_round in [2, 3] else 0.0
        is_rookie_top_50 = 1.0 if draft_pick_num <= 50 else 0.0
        rookie_expected_volume = max(0.0, 160.0 - 0.8 * draft_pick_num)

        is_rb = 1.0 if pos == 'RB' else 0.0
        row_dict = {
            'player_name': r['name'],
            'position': pos,
            'team_predict': team_predict,
            'is_rookie': 1,
            'years_exp': 0,
            'age': 21.0,
            'oline_score': oline_score,
            'team_ypc_ex_qb': team_ypc,
            'draft_pick_num': draft_pick_num,
            'draft_round': draft_round,
            'is_day_1': is_day_1,
            'is_day_2': is_day_2,
            'is_rookie_top_50': is_rookie_top_50,
            'rookie_expected_volume': rookie_expected_volume,
            'prev_passing_yards': 0.0,
            'prev_passing_tds': 0.0,
            'prev_interceptions': 0.0,
            'prev_rushing_yards': 0.0,
            'prev_carries': 0.0,
            'prev_rushing_tds': 0.0,
            'prev_receptions': 0.0,
            'prev_targets': 0.0,
            'prev_receiving_yards': 0.0,
            'prev_receiving_tds': 0.0,
            'prev_fumbles_lost': 0.0,
            'prev_fantasy_points': 0.0,
            'rb_oline_score_inter': is_rb * oline_score,
            'rb_team_ypc_inter': is_rb * team_ypc,
            'wr_oline_score_inter': (1.0 if pos == 'WR' else 0.0) * oline_score,
            'te_oline_score_inter': (1.0 if pos == 'TE' else 0.0) * oline_score
        }
        predict_rows.append(row_dict)

    predict_df = pd.DataFrame(predict_rows)
    return predict_df

def make_draft_list(model, scaler, feature_cols, predict_df):
    """
    Standardizes input features, runs PyTorch model inference,
    clamped to >= 1.0, and returns the sorted draft Cheat Sheet.
    """
    if len(predict_df) == 0:
        print("Warning: Input prediction dataframe is empty!")
        return pd.DataFrame()
        
    # Create position dummies
    df_encoded = pd.get_dummies(predict_df, columns=['position'], prefix='pos')
    for pos in ['QB', 'RB', 'WR', 'TE']:
        col = f'pos_{pos}'
        if col not in df_encoded.columns:
            df_encoded[col] = 0
            
    # Keep only target features in matching order
    X_predict_raw = df_encoded[feature_cols].copy()
    
    # Scale features
    X_predict = scaler.transform(X_predict_raw)
    
    # Model inference
    model.eval()
    with torch.no_grad():
        inputs = torch.tensor(X_predict, dtype=torch.float32)
        preds = model(inputs).numpy().flatten()
        
    # Clamp predicted ADP
    preds = np.clip(preds, 1.0, None)
    
    # Add predictions back
    predict_df['predicted_adp'] = preds
    
    # Sort to form the draft list
    draft_list = predict_df.sort_values(by='predicted_adp').reset_index(drop=True)
    draft_list['draft_rank'] = draft_list.index + 1
    
    return draft_list
