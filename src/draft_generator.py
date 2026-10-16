import torch
import pandas as pd
import numpy as np
from src.config import standardize_team, PREDICT_YEAR, TEAM_OLINE_ADJUSTMENTS, NUM_TEAMS
from src.preprocessing import clean_name

# Dict of major player team movements for the prediction season
# This allows the model to pair a player with their NEW team's offensive line
PLAYER_TEAM_CHANGES = {
    'kyler murray': 'MIN',
    'kylermurray': 'MIN',
    # 'saquon barkley': 'PHI',
    # 'derrick henry': 'BAL',
    # 'austin ekeler': 'WAS',
    # 'stefon diggs': 'WAS',
}

# Adjust ADP based on recent news (August 2026)
PLAYER_NEWS_ADJUSTMENTS = {
    'jahmyr gibbs': -5.0,        # Extension boost
    'luther burden iii': 15.0,   # Groin injury
    'zay flowers': 3.0,          # Quad contusion
    'mike evans': 5.0,           # Quad injury, cautious
    'malik nabers': -5.0,        # Back to practice
    'daniel jones': -10.0,       # Practicing full, looking strong
    'josh downs': 2.0,           # Minor groin issue
    'tua tagovailoa': -5.0,      # Early-season starter
    'jaxson dart': -15.0,        # Breakout candidate
    'jakobi lane': -10.0,        # Buzz
    'pat bryant': -10.0,         # Standout
    'seth mcgowan': -5.0         # RB2 behind starter
}

from src.preprocessing import load_draft_picks, load_depth_charts, load_depth_charts_teams, compute_coach_metrics
from src.def_model import generate_def_projections

# List of true incoming 2026 NFL rookies (entering their first season in Sept 2026)
ROOKIE_PROSPECTS_2026 = [
    # Seattle Seahawks 2026 Rookie Spotlight
    {'name': 'Jeremiah Price', 'pos': 'RB', 'team': 'SEA', 'pick': 32, 'round': 1, 'college': 'Notre Dame'},
    
    # Notre Dame 2026 Drafted Prospects
    {'name': 'Jeremiyah Love', 'pos': 'RB', 'team': 'ARI', 'pick': 15, 'round': 1, 'college': 'Notre Dame'},
    {'name': 'Aneyas Williams', 'pos': 'RB', 'team': 'DET', 'pick': 120, 'round': 4, 'college': 'Notre Dame'},
    {'name': 'Kedren Young', 'pos': 'RB', 'team': 'GB', 'pick': 135, 'round': 5, 'college': 'Notre Dame'},
    {'name': 'Mitchell Evans', 'pos': 'TE', 'team': 'GB', 'pick': 85, 'round': 3, 'college': 'Notre Dame'},
    
    # 2026 NFL Draft Notable Selections
    {'name': 'Fernando Mendoza', 'pos': 'QB', 'team': 'LV', 'pick': 1, 'round': 1, 'college': 'Indiana'},
    {'name': 'KC Concepcion', 'pos': 'WR', 'team': 'CLE', 'pick': 45, 'round': 2, 'college': 'NC State'},
    {'name': 'Denzel Boston', 'pos': 'WR', 'team': 'CLE', 'pick': 75, 'round': 3, 'college': 'Washington'},
    {'name': 'David Bailey', 'pos': 'TE', 'team': 'NYJ', 'pick': 10, 'round': 1, 'college': 'Texas Tech'}, # Listed as EDGE originally but putting TE for fantasy
    
    # Returning/Injured Players with no Y-1 stats to be evaluated via draft capital
    {'name': 'Jonathon Brooks', 'pos': 'RB', 'team': 'CAR', 'pick': 46, 'round': 2, 'college': 'Texas'}
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
    depth_map = load_depth_charts(PREDICT_YEAR)
    depth_team_map, depth_name_only = load_depth_charts_teams(PREDICT_YEAR)
    coach_map = compute_coach_metrics(player_stats_df)
    
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
        
        # Determine their team for PREDICT_YEAR:
        # Priority:
        # 1. Transfer overrides in PLAYER_TEAM_CHANGES
        # 2. Official NFL depth chart 2026 team
        # 3. Fallback to stats_prev (2025 team)
        team_predict = (
            PLAYER_TEAM_CHANGES.get(cname) or 
            depth_team_map.get((cname, pos)) or 
            depth_name_only.get(cname) or 
            row['std_team']
        )
        
        # Get their age and experience for PREDICT_YEAR
        roster_key = (stats_year, cname, pos)
        years_exp = 1  # default
        age = 26.0  # default
        height = 73.0
        weight = 215.0
        
        roster_info = roster_map.get(roster_key, None)
        if roster_info:
            years_exp = roster_info['years_exp'] + 1
            height = roster_info['height']
            weight = roster_info['weight']
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
            height = 73.0
            weight = 215.0
            
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
        
        # Depth Chart Features
        pos_rank = depth_map.get((cname, pos), 99)
        is_team_wr1 = 1.0 if (pos == 'WR' and pos_rank == 1) else 0.0
        is_team_wr2 = 1.0 if (pos == 'WR' and pos_rank == 2) else 0.0
        is_team_rb1 = 1.0 if (pos == 'RB' and pos_rank == 1) else 0.0
        
        # Coach Features
        coach_info = coach_map.get((PREDICT_YEAR, team_predict), {})
        coach_age = coach_info.get('coach_age', 50.0)
        coach_player_age_gap = coach_age - age
        
        # Create row
        is_rb = 1.0 if pos == 'RB' else 0.0
        row_dict = {
            'player_name': name,
            'position': pos,
            'team_predict': team_predict,
            'is_rookie': 0,
            'years_exp': years_exp,
            'age': age,
            'height': height,
            'weight': weight,
            'oline_score': oline_score,
            'team_ypc_ex_qb': team_ypc,
            # Pillar 1 Features
            'draft_pick_num': draft_pick_num,
            'draft_round': draft_round,
            'is_day_1': is_day_1,
            'is_day_2': is_day_2,
            'is_rookie_top_50': is_rookie_top_50,
            'rookie_expected_volume': rookie_expected_volume,
            # Depth Chart Features
            'is_team_wr1': is_team_wr1,
            'is_team_wr2': is_team_wr2,
            'is_team_rb1': is_team_rb1,
            # Coach History & Age Features
            'coach_age': coach_age,
            'is_young_innovator': coach_info.get('is_young_innovator', 0.0),
            'is_old_school': coach_info.get('is_old_school', 0.0),
            'coach_player_age_gap': coach_player_age_gap,
            'coach_experience': coach_info.get('coach_experience', 0),
            'coach_win_pct': coach_info.get('coach_win_pct', 0.5),
            'franchise_tenure': coach_info.get('franchise_tenure', 0),
            'coach_bellcow_index': coach_info.get('coach_bellcow_index', 0.5),
            'coach_pass_to_rb_share': coach_info.get('coach_pass_to_rb_share', 0.18),
            'coach_te_target_share': coach_info.get('coach_te_target_share', 0.20),
            'first_year_hc_flag': coach_info.get('first_year_hc_flag', 0.0),
            'coach_pass_rate': coach_info.get('coach_pass_to_rb_share', 0.5), # Proxy if needed
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

        pos_rank = depth_map.get((r_cname, pos), 99)
        is_team_wr1 = 1.0 if (pos == 'WR' and pos_rank == 1) else 0.0
        is_team_wr2 = 1.0 if (pos == 'WR' and pos_rank == 2) else 0.0
        is_team_rb1 = 1.0 if (pos == 'RB' and pos_rank == 1) else 0.0

        coach_info = coach_map.get((PREDICT_YEAR, team_predict), {})
        coach_age = coach_info.get('coach_age', 50.0)
        coach_player_age_gap = coach_age - 21.0

        is_rb = 1.0 if pos == 'RB' else 0.0
        row_dict = {
            'player_name': r['name'],
            'position': pos,
            'team_predict': team_predict,
            'is_rookie': 1,
            'years_exp': 0,
            'age': 21.0,
            'height': 73.0,
            'weight': 215.0,
            'oline_score': oline_score,
            'team_ypc_ex_qb': team_ypc,
            'draft_pick_num': draft_pick_num,
            'draft_round': draft_round,
            'is_day_1': is_day_1,
            'is_day_2': is_day_2,
            'is_rookie_top_50': is_rookie_top_50,
            'rookie_expected_volume': rookie_expected_volume,
            'is_team_wr1': is_team_wr1,
            'is_team_wr2': is_team_wr2,
            'is_team_rb1': is_team_rb1,
            'coach_age': coach_age,
            'is_young_innovator': coach_info.get('is_young_innovator', 0.0),
            'is_old_school': coach_info.get('is_old_school', 0.0),
            'coach_player_age_gap': coach_player_age_gap,
            'coach_experience': coach_info.get('coach_experience', 0),
            'coach_win_pct': coach_info.get('coach_win_pct', 0.5),
            'franchise_tenure': coach_info.get('franchise_tenure', 0),
            'coach_bellcow_index': coach_info.get('coach_bellcow_index', 0.5),
            'coach_pass_to_rb_share': coach_info.get('coach_pass_to_rb_share', 0.18),
            'coach_te_target_share': coach_info.get('coach_te_target_share', 0.20),
            'first_year_hc_flag': coach_info.get('first_year_hc_flag', 0.0),
            'coach_pass_rate': coach_info.get('coach_pass_to_rb_share', 0.5),
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

def make_draft_list(model_adp, model_pts, scaler, feature_cols, predict_df):
    """
    Standardizes input features, runs PyTorch model inference for ADP and Points,
    and returns the sorted draft Cheat Sheet.
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
    model_adp.eval()
    model_pts.eval()
    with torch.no_grad():
        inputs = torch.tensor(X_predict, dtype=torch.float32)
        preds_adp = model_adp(inputs).numpy().flatten()
        preds_pts = model_pts(inputs).numpy().flatten()
        
    # Clamp predictions
    preds_adp = np.clip(preds_adp, 1.0, None)
    preds_pts = np.clip(preds_pts, 0.0, None)
    
    # Add predictions back
    predict_df['predicted_adp'] = preds_adp
    predict_df['predicted_pts'] = preds_pts
    
    # Apply News Adjustments to ADP
    for idx, row in predict_df.iterrows():
        cname = clean_name(row['player_name'])
        if cname in PLAYER_NEWS_ADJUSTMENTS:
            predict_df.at[idx, 'predicted_adp'] += PLAYER_NEWS_ADJUSTMENTS[cname]
    
    # Calculate VORP (Value Over Replacement Player)
    baseline_ranks = {
        'QB': int(1.5 * NUM_TEAMS), 
        'RB': int(4.0 * NUM_TEAMS), 
        'WR': int(4.5 * NUM_TEAMS), 
        'TE': int(1.5 * NUM_TEAMS)
    }
    vorp_list = []
    
    for pos in ['QB', 'RB', 'WR', 'TE']:
        pos_df = predict_df[predict_df['position'] == pos].copy()
        if len(pos_df) > 0:
            pos_df = pos_df.sort_values(by='predicted_pts', ascending=False)
            rank = min(baseline_ranks[pos], len(pos_df)) - 1 # 0-indexed
            baseline_pts = pos_df.iloc[rank]['predicted_pts']
            pos_df['vorp'] = pos_df['predicted_pts'] - baseline_pts
            vorp_list.append(pos_df)
            
    if vorp_list:
        predict_df = pd.concat(vorp_list)
    else:
        predict_df['vorp'] = 0.0
    
    # Add Bayesian DEF Projections
    def_projections = generate_def_projections(PREDICT_YEAR)
    def_rows = []
    for team, data in def_projections.items():
        def_rows.append({
            'player_name': team,
            'position': 'DEF',
            'team_predict': team.replace(' DEF', ''),
            'predicted_adp': 160.0,
            'predicted_pts': float(data['points']),
            'vorp': float(data['vorp'])
        })
    
    if def_rows:
        df_def = pd.DataFrame(def_rows)
        for col in predict_df.columns:
            if col not in df_def.columns:
                df_def[col] = 0.0
        predict_df = pd.concat([predict_df, df_def], ignore_index=True)
    
    # Sort to form the draft list
    draft_list = predict_df.sort_values(by='predicted_adp').reset_index(drop=True)
    draft_list['draft_rank'] = draft_list.index + 1
    
    return draft_list
