import os
import json
import re
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from src.config import START_YEAR, END_YEAR, PREDICT_YEAR, CACHE_DIR, standardize_team

def clean_name(name):
    """Standardizes player names for robust cross-dataset matching."""
    if not name:
        return ""
    name = str(name).lower()
    # Remove suffixes (jr, sr, iii, ii, iv, etc.)
    name = re.sub(r'\b(jr|sr|iii|ii|iv|v|esq)\b', '', name)
    # Remove all non-alphanumeric characters
    name = re.sub(r'[^a-z0-9]', '', name)
    return name.strip()

def compute_oline_scores(player_stats_df):
    """
    Computes a seasonal composite Offensive Line Quality Score for each NFL team.
    Run blocking score: team yards per carry (YPC) excluding QB runs.
    Pass blocking score: team sack rate (sacks / (pass attempts + sacks)).
    Both are standardized (Z-scored) per season, and the composite is their average.
    """
    print("Computing offensive line metrics...")
    reg_df = player_stats_df.copy()
    if 'season_type' in reg_df.columns:
        reg_df = reg_df[reg_df['season_type'] == 'REG']
    
    # Standardize team names
    reg_df['std_team'] = reg_df['recent_team'].apply(standardize_team)
    
    # 1. RUN BLOCKING: Sum rushing stats for non-QBs (RB, WR, TE)
    non_qb_mask = ~reg_df['position'].isin(['QB'])
    team_rush = reg_df[non_qb_mask].groupby(['std_team', 'season']).agg(
        total_rush_yards=('rushing_yards', 'sum'),
        total_carries=('carries', 'sum')
    ).reset_index()
    
    team_rush['team_ypc_ex_qb'] = team_rush['total_rush_yards'] / team_rush['total_carries'].replace(0, 1)
    # Fill any NaNs
    team_rush['team_ypc_ex_qb'] = team_rush['team_ypc_ex_qb'].fillna(4.0)
    
    # 2. PASS BLOCKING: Sum passing/sack stats for the team
    team_pass = reg_df.groupby(['std_team', 'season']).agg(
        total_sacks=('sacks', 'sum'),
        total_pass_attempts=('attempts', 'sum')
    ).reset_index()
    
    team_pass['team_sack_rate'] = team_pass['total_sacks'] / (team_pass['total_pass_attempts'] + team_pass['total_sacks']).replace(0, 1)
    team_pass['team_sack_rate'] = team_pass['team_sack_rate'].fillna(0.06) # standard average
    
    # Merge run and pass blocking
    oline_df = pd.merge(team_rush, team_pass, on=['std_team', 'season'], how='outer')
    
    # Standardize per season
    oline_df['oline_score'] = 0.0
    oline_df['team_ypc_z'] = 0.0
    oline_df['team_sack_z'] = 0.0
    
    for season in oline_df['season'].unique():
        season_mask = oline_df['season'] == season
        season_data = oline_df[season_mask]
        
        if len(season_data) > 1:
            # Yards per carry (higher is better)
            ypc_mean = season_data['team_ypc_ex_qb'].mean()
            ypc_std = season_data['team_ypc_ex_qb'].std() or 1.0
            oline_df.loc[season_mask, 'team_ypc_z'] = (season_data['team_ypc_ex_qb'] - ypc_mean) / ypc_std
            
            # Sack rate (lower is better, so negate the Z-score)
            sack_mean = season_data['team_sack_rate'].mean()
            sack_std = season_data['team_sack_rate'].std() or 1.0
            oline_df.loc[season_mask, 'team_sack_z'] = - (season_data['team_sack_rate'] - sack_mean) / sack_std
            
    # Composite score
    oline_df['oline_score'] = 0.5 * oline_df['team_ypc_z'] + 0.5 * oline_df['team_sack_z']
    
    # Return a clean mapping dictionary: {(season, team): oline_score, team_ypc_ex_qb}
    oline_map = {}
    for _, row in oline_df.iterrows():
        oline_map[(row['season'], row['std_team'])] = {
            'oline_score': row['oline_score'],
            'team_ypc_ex_qb': row['team_ypc_ex_qb']
        }
    return oline_map

def load_roster_data():
    """Loads all cached roster files and creates a map of {(season, player_name, position): years_exp}."""
    print("Loading roster data...")
    roster_map = {}
    
    # Start looking from START_YEAR-1 to PREDICT_YEAR - 1
    for year in range(START_YEAR - 1, PREDICT_YEAR):
        filepath = os.path.join(CACHE_DIR, f"roster_{year}.parquet")
        if os.path.exists(filepath):
            try:
                df = pd.read_parquet(filepath)
                # Keep useful columns
                df = df[['season', 'full_name', 'position', 'years_exp', 'birth_date']].dropna(subset=['full_name', 'position'])
                for _, row in df.iterrows():
                    name_key = clean_name(row['full_name'])
                    pos_key = str(row['position']).upper()
                    
                    # Convert experience
                    try:
                        exp = int(row['years_exp'])
                    except:
                        exp = 0
                        
                    roster_map[(row['season'], name_key, pos_key)] = {
                        'years_exp': exp,
                        'birth_date': row['birth_date']
                    }
            except Exception as e:
                print(f"Error loading roster for {year}: {e}")
                
    return roster_map

def load_aggregate_player_stats():
    """Loads player stats parquet, aggregates per-season, and calculates Half-PPR fantasy points."""
    print("Loading and aggregating player statistics...")
    filepath = os.path.join(CACHE_DIR, "player_stats.parquet")
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Player stats file not found at {filepath}. Run data_fetcher first.")
        
    df = pd.read_parquet(filepath)
    df = df[df['season_type'] == 'REG'].copy()
    
    # Use player_display_name as the primary name for matching, falling back to player_name
    if 'player_display_name' in df.columns:
        df['player_name'] = df['player_display_name'].fillna(df['player_name']).fillna("Unknown")
    else:
        df['player_name'] = df['player_name'].fillna("Unknown")
    df['recent_team'] = df['recent_team'].fillna("UNK")
    df['position'] = df['position'].fillna("UNK")
    
    # Fill NaNs in stats columns with 0
    stats_cols = [
        'attempts', 'sacks', 'passing_yards', 'passing_tds', 'interceptions', 
        'rushing_yards', 'carries', 'rushing_tds',
        'rushing_fumbles_lost', 'receiving_fumbles_lost', 'sack_fumbles_lost',
        'receptions', 'targets', 'receiving_yards', 'receiving_tds'
    ]
    for col in stats_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0.0)
            
    # Calculate fumbles lost
    fumbles_lost = 0
    for col in ['rushing_fumbles_lost', 'receiving_fumbles_lost', 'sack_fumbles_lost']:
        if col in df.columns:
            fumbles_lost += df[col]
    df['fumbles_lost'] = fumbles_lost
    
    # Calculate Half-PPR fantasy points
    df['fantasy_points_half_ppr'] = (
        df['passing_yards'] * 0.04 + 
        df['passing_tds'] * 4.0 - 
        df['interceptions'] * 2.0 + 
        df['rushing_yards'] * 0.1 + 
        df['rushing_tds'] * 6.0 + 
        df['receptions'] * 0.5 + 
        df['receiving_yards'] * 0.1 + 
        df['receiving_tds'] * 6.0 - 
        df['fumbles_lost'] * 2.0
    )
    
    # Group by player and season
    agg_df = df.groupby(['player_id', 'player_name', 'position', 'recent_team', 'season']).agg(
        attempts=('attempts', 'sum'),
        sacks=('sacks', 'sum'),
        passing_yards=('passing_yards', 'sum'),
        passing_tds=('passing_tds', 'sum'),
        interceptions=('interceptions', 'sum'),
        rushing_yards=('rushing_yards', 'sum'),
        carries=('carries', 'sum'),
        rushing_tds=('rushing_tds', 'sum'),
        receptions=('receptions', 'sum'),
        targets=('targets', 'sum'),
        receiving_yards=('receiving_yards', 'sum'),
        receiving_tds=('receiving_tds', 'sum'),
        fumbles_lost=('fumbles_lost', 'sum'),
        fantasy_points=('fantasy_points_half_ppr', 'sum')
    ).reset_index()
    
    return agg_df

def load_draft_picks():
    """
    Downloads and caches nflverse draft picks data.
    Returns: draft_map: {(clean_name, pos): {'pick': int, 'round': int, 'college': str}}
    """
    cache_path = os.path.join(CACHE_DIR, "draft_picks.parquet")
    if not os.path.exists(cache_path):
        try:
            url = 'https://github.com/nflverse/nflverse-data/releases/download/draft_picks/draft_picks.parquet'
            df = pd.read_parquet(url)
            df.to_parquet(cache_path)
        except Exception as e:
            print(f"Warning: Could not download draft_picks.parquet: {e}")
            return {}
    else:
        df = pd.read_parquet(cache_path)
        
    df['clean_name'] = df['pfr_player_name'].apply(clean_name)
    draft_map = {}
    for _, row in df.iterrows():
        p_name = row['clean_name']
        c_pos = str(row['category']).upper()
        if c_pos in ['QB', 'RB', 'WR', 'TE']:
            draft_map[(p_name, c_pos)] = {
                'pick': int(row['pick']) if pd.notnull(row['pick']) else 250,
                'round': int(row['round']) if pd.notnull(row['round']) else 8,
                'college': str(row['college']) if pd.notnull(row['college']) else 'Other'
            }
    return draft_map

def prepare_ml_dataset():
    """
    Builds the complete aligned dataset:
    For a given draft year Y (2008-2025):
    - Target: player's ADP in year Y.
    - Features: player's stats in year Y-1, player's age/exp in year Y,
      new team's O-line metrics, and College Draft Capital (Pillar 1).
    """
    player_stats_df = load_aggregate_player_stats()
    oline_map = compute_oline_scores(player_stats_df)
    roster_map = load_roster_data()
    draft_map = load_draft_picks()
    
    # Clean player names in stats for matching
    player_stats_df['clean_name'] = player_stats_df['player_name'].apply(clean_name)
    player_stats_df['std_team'] = player_stats_df['recent_team'].apply(standardize_team)
    
    # Create player stats map: {(season, clean_name, position): stats_dict}
    stats_map = {}
    for _, row in player_stats_df.iterrows():
        stats_map[(row['season'], row['clean_name'], row['position'])] = row.to_dict()
        
    dataset_rows = []
    
    # Loop over draft years Y (2008 to END_YEAR)
    start_adp_year = max(2008, START_YEAR)
    for year in range(start_adp_year, END_YEAR + 1):
        adp_filepath = os.path.join(CACHE_DIR, f"adp_{year}.json")
        if not os.path.exists(adp_filepath):
            continue
            
        with open(adp_filepath, 'r', encoding='utf-8') as f:
            adp_data = json.load(f)
            
        # Parse players from FFC ADP
        adp_players = adp_data.get('players', [])
        for p in adp_players:
            pos = str(p.get('position')).upper()
            if pos not in ['QB', 'RB', 'WR', 'TE']:
                continue
                
            name = p.get('name')
            adp_val = p.get('adp')
            team_ffc = standardize_team(p.get('team'))
            
            clean_pname = clean_name(name)
            
            # Match with Y-1 stats
            prev_season = year - 1
            stats_key = (prev_season, clean_pname, pos)
            
            # Match with Y roster
            roster_key = (year, clean_pname, pos)
            
            # Get experience and age
            years_exp = 0
            age = 26.0  # default median
            
            roster_info = roster_map.get(roster_key, roster_map.get((prev_season, clean_pname, pos), None))
            if roster_info:
                years_exp = roster_info['years_exp']
                birth_date = roster_info['birth_date']
                if birth_date:
                    try:
                        birth_year = pd.to_datetime(birth_date).year
                        age = year - birth_year
                    except:
                        pass
            
            # If not in stats_map, they had no stats last year (e.g. Rookie or injured)
            is_rookie = 0
            if stats_key in stats_map:
                prev_stats = stats_map[stats_key]
            else:
                prev_stats = {
                    'passing_yards': 0.0, 'passing_tds': 0.0, 'interceptions': 0.0,
                    'rushing_yards': 0.0, 'carries': 0.0, 'rushing_tds': 0.0,
                    'receptions': 0.0, 'targets': 0.0, 'receiving_yards': 0.0,
                    'receiving_tds': 0.0, 'fumbles_lost': 0.0, 'fantasy_points': 0.0
                }
                if years_exp <= 1:
                    is_rookie = 1
                    
            # Get team's O-line score in year Y (since they will play behind this line)
            oline_info = oline_map.get((year, team_ffc), {'oline_score': 0.0, 'team_ypc_ex_qb': 4.0})
            oline_score = oline_info['oline_score']
            team_ypc = oline_info['team_ypc_ex_qb']

            # Pillar 1: College Draft Capital Features
            draft_info = draft_map.get((clean_pname, pos), {'pick': 250, 'round': 8, 'college': 'Other'})
            draft_pick_num = draft_info['pick']
            draft_round = draft_info['round']
            is_day_1 = 1.0 if draft_round == 1 else 0.0
            is_day_2 = 1.0 if draft_round in [2, 3] else 0.0
            is_rookie_top_50 = 1.0 if (is_rookie == 1 and draft_pick_num <= 50) else 0.0
            rookie_expected_volume = max(0.0, 160.0 - 0.8 * draft_pick_num) if is_rookie == 1 else 0.0
            
            # Create feature row
            row_dict = {
                'player_name': name,
                'position': pos,
                'year': year,
                'adp': adp_val,
                'is_rookie': is_rookie,
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
                # Prev season stats
                'prev_passing_yards': prev_stats['passing_yards'],
                'prev_passing_tds': prev_stats['passing_tds'],
                'prev_interceptions': prev_stats['interceptions'],
                'prev_rushing_yards': prev_stats['rushing_yards'],
                'prev_carries': prev_stats['carries'],
                'prev_rushing_tds': prev_stats['rushing_tds'],
                'prev_receptions': prev_stats['receptions'],
                'prev_targets': prev_stats['targets'],
                'prev_receiving_yards': prev_stats['receiving_yards'],
                'prev_receiving_tds': prev_stats['receiving_tds'],
                'prev_fumbles_lost': prev_stats['fumbles_lost'],
                'prev_fantasy_points': prev_stats['fantasy_points']
            }
            
            # Create Running Back Specific O-Line Interaction features
            is_rb = 1.0 if pos == 'RB' else 0.0
            row_dict['rb_oline_score_inter'] = is_rb * oline_score
            row_dict['rb_team_ypc_inter'] = is_rb * team_ypc
            
            # Interaction features for WRs and TEs
            row_dict['wr_oline_score_inter'] = (1.0 if pos == 'WR' else 0.0) * oline_score
            row_dict['te_oline_score_inter'] = (1.0 if pos == 'TE' else 0.0) * oline_score
            
            dataset_rows.append(row_dict)
            
    dataset_df = pd.DataFrame(dataset_rows)
    print(f"Created complete dataset with {len(dataset_df)} rows and {len(dataset_df.columns)} columns.")
    return dataset_df

def split_and_scale_data(df):
    """
    Splits the data chronologically:
    - Train: 2008 to 2023
    - Val: 2024 (latest complete year with ADP)
    Scales numerical features and returns split datasets.
    """
    # Features list (excluding metadata and target)
    feature_cols = [
        'is_rookie', 'years_exp', 'age', 'oline_score', 'team_ypc_ex_qb',
        # Pillar 1 Features
        'draft_pick_num', 'draft_round', 'is_day_1', 'is_day_2',
        'is_rookie_top_50', 'rookie_expected_volume',
        'prev_passing_yards', 'prev_passing_tds', 'prev_interceptions',
        'prev_rushing_yards', 'prev_carries', 'prev_rushing_tds',
        'prev_receptions', 'prev_targets', 'prev_receiving_yards',
        'prev_receiving_tds', 'prev_fumbles_lost', 'prev_fantasy_points',
        'rb_oline_score_inter', 'rb_team_ypc_inter',
        'wr_oline_score_inter', 'te_oline_score_inter',
        # One-hot encoded positions
        'pos_QB', 'pos_RB', 'pos_WR', 'pos_TE'
    ]
    
    # One-hot encode position
    df_encoded = pd.get_dummies(df, columns=['position'], prefix='pos')
    # Make sure all position columns exist
    for pos in ['QB', 'RB', 'WR', 'TE']:
        col = f'pos_{pos}'
        if col not in df_encoded.columns:
            df_encoded[col] = 0
            
    # Chronological Split
    train_mask = df_encoded['year'] < 2024
    val_mask = df_encoded['year'] == 2024
    
    train_data = df_encoded[train_mask]
    val_data = df_encoded[val_mask]
    
    X_train_raw = train_data[feature_cols].copy()
    y_train = train_data['adp'].values
    
    X_val_raw = val_data[feature_cols].copy()
    y_val = val_data['adp'].values
    
    # Scale features
    scaler = StandardScaler()
    
    # We only fit on train to avoid data leakage
    X_train = scaler.fit_transform(X_train_raw)
    X_val = scaler.transform(X_val_raw)
    
    return X_train, y_train, X_val, y_val, scaler, feature_cols

if __name__ == "__main__":
    df = prepare_ml_dataset()
    if len(df) > 0:
        X_train, y_train, X_val, y_val, scaler, feature_cols = split_and_scale_data(df)
        print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
        print(f"X_val shape: {X_val.shape}, y_val shape: {y_val.shape}")
