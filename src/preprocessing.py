import os
import json
import re
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from src.config import START_YEAR, END_YEAR, PREDICT_YEAR, CACHE_DIR, standardize_team, LEAGUE_CONFIG

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
                df = df[['season', 'full_name', 'position', 'years_exp', 'birth_date', 'height', 'weight']].dropna(subset=['full_name', 'position'])
                for _, row in df.iterrows():
                    name_key = clean_name(row['full_name'])
                    pos_key = str(row['position']).upper()
                    
                    # Convert experience
                    try:
                        exp = int(row['years_exp'])
                    except:
                        exp = 0
                        
                    # Convert height and weight
                    try:
                        h = float(row['height'])
                    except:
                        h = 73.0
                    try:
                        w = float(row['weight'])
                    except:
                        w = 215.0
                        
                    roster_map[(row['season'], name_key, pos_key)] = {
                        'years_exp': exp,
                        'birth_date': row['birth_date'],
                        'height': h,
                        'weight': w
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
    
    # Calculate fantasy points based on LEAGUE_CONFIG
    reception_points = 1.0 if LEAGUE_CONFIG == '12_team_full_ppr' else 0.5
    
    df['calculated_fantasy_points'] = (
        df['passing_yards'] * 0.04 + 
        df['passing_tds'] * 4.0 - 
        df['interceptions'] * 2.0 + 
        df['rushing_yards'] * 0.1 + 
        df['rushing_tds'] * 6.0 + 
        df['receptions'] * reception_points + 
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
        fantasy_points=('calculated_fantasy_points', 'sum')
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

def load_depth_charts(year):
    """
    Downloads and caches nflverse depth charts for the given year.
    Returns: depth_map: {(clean_name, pos): pos_rank}
    """
    cache_path = os.path.join(CACHE_DIR, f"depth_charts_{year}.parquet")
    if not os.path.exists(cache_path):
        try:
            url = f'https://github.com/nflverse/nflverse-data/releases/download/depth_charts/depth_charts_{year}.parquet'
            df = pd.read_parquet(url)
            df.to_parquet(cache_path)
        except Exception as e:
            # Silently fail if year doesn't exist
            return {}
    else:
        df = pd.read_parquet(cache_path)
        
    name_col = 'player_name' if 'player_name' in df.columns else 'full_name'
    if name_col not in df.columns:
        # Fallback if no name column exists
        return {}
    df['clean_name'] = df[name_col].apply(clean_name)
    pos_col = 'pos_abb' if 'pos_abb' in df.columns else ('position' if 'position' in df.columns else 'pos')
    if pos_col not in df.columns:
        return {}
    
    # Filter to latest available dt (pre-season kickoff state)
    if 'dt' in df.columns:
        df = df.sort_values('dt').drop_duplicates(subset=['clean_name', pos_col], keep='last')
        
    depth_map = {}
    for _, row in df.iterrows():
        p_name = row['clean_name']
        pos = str(row[pos_col]).upper()
        if pos in ['WR', 'RB', 'TE', 'QB']:
            try:
                depth_map[(p_name, pos)] = int(row['pos_rank'])
            except:
                pass
    return depth_map


def load_depth_charts_teams(year):
    """
    Loads nflverse depth charts for the given year and extracts player teams.
    Returns: depth_team_map: {(clean_name, pos): team_abbr}, name_only_map: {clean_name: team_abbr}
    """
    cache_path = os.path.join(CACHE_DIR, f"depth_charts_{year}.parquet")
    if not os.path.exists(cache_path):
        return {}, {}
    df = pd.read_parquet(cache_path)
    name_col = 'player_name' if 'player_name' in df.columns else 'full_name'
    if name_col not in df.columns:
        return {}, {}
    df['clean_name'] = df[name_col].apply(clean_name)
    pos_col = 'pos_abb' if 'pos_abb' in df.columns else ('position' if 'position' in df.columns else 'pos')
    if pos_col not in df.columns:
        return {}, {}
    
    # Sort by dt to keep the latest pre-season depth chart state
    if 'dt' in df.columns:
        df = df.sort_values('dt').drop_duplicates(subset=['clean_name', pos_col], keep='last')
        
    depth_team_map = {}
    name_only_map = {}
    for _, row in df.iterrows():
        p_name = row['clean_name']
        pos = str(row[pos_col]).upper()
        team_val = standardize_team(row.get('team'))
        if team_val and team_val != 'UNK':
            depth_team_map[(p_name, pos)] = team_val
            name_only_map[p_name] = team_val
            
    return depth_team_map, name_only_map


def compute_coach_metrics(player_stats_df):
    """
    Computes coach history, demographic, and playcalling features.
    Returns: coach_map: {(season, team): coach_features_dict}
    """
    print("Computing coach metrics...")
    games_path = os.path.join(CACHE_DIR, "games.parquet")
    if not os.path.exists(games_path):
        print("Warning: games.parquet not found. Returning empty coach metrics.")
        return {}
        
    games_df = pd.read_parquet(games_path)
    
    # Load coach ages
    coach_birth_years = {}
    if os.path.exists("coach_birth_years.json"):
        with open("coach_birth_years.json", "r") as f:
            coach_birth_years = json.load(f)
            
    # Map coaches to team/season
    home_coaches = games_df[['season', 'home_team', 'home_coach']].rename(columns={'home_team': 'team', 'home_coach': 'coach'})
    away_coaches = games_df[['season', 'away_team', 'away_coach']].rename(columns={'away_team': 'team', 'away_coach': 'coach'})
    team_coaches = pd.concat([home_coaches, away_coaches]).drop_duplicates(subset=['season', 'team', 'coach'])
    
    # For a team with multiple coaches in a season, just take the first one for simplicity
    team_coaches = team_coaches.groupby(['season', 'team']).first().reset_index()
    team_coaches['std_team'] = team_coaches['team'].apply(standardize_team)
    
    # Pre-calculate Coach History
    # Cumulative Win Pct and Experience
    games_df['home_win'] = (games_df['home_score'] > games_df['away_score']).astype(int)
    games_df['away_win'] = (games_df['away_score'] > games_df['home_score']).astype(int)
    
    coach_history = []
    for coach in team_coaches['coach'].unique():
        if pd.isna(coach): continue
        c_games = games_df[(games_df['home_coach'] == coach) | (games_df['away_coach'] == coach)].sort_values('season')
        cumulative_wins = 0
        cumulative_games = 0
        seasons_active = set()
        
        for idx, row in c_games.iterrows():
            is_home = (row['home_coach'] == coach)
            won = row['home_win'] if is_home else row['away_win']
            season = row['season']
            
            # Record state before the game
            coach_history.append({
                'coach': coach,
                'season': season,
                'wins_before': cumulative_wins,
                'games_before': cumulative_games,
                'seasons_active_before': len(seasons_active)
            })
            
            cumulative_wins += won
            cumulative_games += 1
            seasons_active.add(season)
            
    if not coach_history:
        return {}
        
    ch_df = pd.DataFrame(coach_history).groupby(['coach', 'season']).first().reset_index()
    ch_df['coach_win_pct'] = ch_df['wins_before'] / ch_df['games_before'].replace(0, 1)
    ch_df['coach_experience'] = ch_df['seasons_active_before']
    ch_df['first_year_hc_flag'] = (ch_df['coach_experience'] == 0).astype(float)
    
    # Tenure
    team_coaches = pd.merge(team_coaches, ch_df, on=['coach', 'season'], how='left')
    team_coaches = team_coaches.sort_values(['team', 'season'])
    
    tenure_list = []
    for team, group in team_coaches.groupby('team'):
        tenure = 0
        prev_coach = None
        for idx, row in group.iterrows():
            if row['coach'] == prev_coach:
                tenure += 1
            else:
                tenure = 0
            tenure_list.append(tenure)
            prev_coach = row['coach']
    team_coaches['franchise_tenure'] = tenure_list
    
    # Calculate Playcalling Preferences from previous season player_stats
    reg_df = player_stats_df.copy()
    reg_df['std_team'] = reg_df['recent_team'].apply(standardize_team)
    
    # Group by team and season
    team_stats = reg_df.groupby(['std_team', 'season']).agg(
        total_targets=('targets', 'sum'),
        total_rb_targets=('targets', lambda x: x[reg_df.loc[x.index, 'position'] == 'RB'].sum()),
        total_te_targets=('targets', lambda x: x[reg_df.loc[x.index, 'position'] == 'TE'].sum()),
        total_rb_touches=('carries', lambda x: (x + reg_df.loc[x.index, 'targets'])[reg_df.loc[x.index, 'position'] == 'RB'].sum()),
        max_rb_touches=('carries', lambda x: (x + reg_df.loc[x.index, 'targets'])[reg_df.loc[x.index, 'position'] == 'RB'].max())
    ).reset_index()
    
    team_stats['coach_pass_to_rb_share'] = team_stats['total_rb_targets'] / team_stats['total_targets'].replace(0, 1)
    team_stats['coach_te_target_share'] = team_stats['total_te_targets'] / team_stats['total_targets'].replace(0, 1)
    team_stats['coach_bellcow_index'] = team_stats['max_rb_touches'] / team_stats['total_rb_touches'].replace(0, 1)
    
    # Shift team stats to the next year (so year Y gets Y-1 stats)
    team_stats['season'] = team_stats['season'] + 1
    
    # Merge with team coaches
    team_coaches = pd.merge(team_coaches, team_stats[['std_team', 'season', 'coach_pass_to_rb_share', 'coach_te_target_share', 'coach_bellcow_index']], 
                            on=['std_team', 'season'], how='left')
                            
    # Fill NAs
    team_coaches['coach_pass_to_rb_share'] = team_coaches['coach_pass_to_rb_share'].fillna(0.18)
    team_coaches['coach_te_target_share'] = team_coaches['coach_te_target_share'].fillna(0.20)
    team_coaches['coach_bellcow_index'] = team_coaches['coach_bellcow_index'].fillna(0.5)
    
    # Compute demographics
    def get_coach_age(row):
        coach = row['coach']
        season = row['season']
        birth_year = coach_birth_years.get(coach)
        if birth_year:
            return season - birth_year
        return 50 # median
        
    team_coaches['coach_age'] = team_coaches.apply(get_coach_age, axis=1)
    team_coaches['is_young_innovator'] = (team_coaches['coach_age'] < 40).astype(float)
    team_coaches['is_old_school'] = (team_coaches['coach_age'] > 60).astype(float)
    
    coach_map = {}
    for _, row in team_coaches.iterrows():
        coach_map[(row['season'], row['std_team'])] = {
            'coach_age': row['coach_age'],
            'is_young_innovator': row['is_young_innovator'],
            'is_old_school': row['is_old_school'],
            'coach_experience': row.get('coach_experience', 0),
            'coach_win_pct': row.get('coach_win_pct', 0.5),
            'franchise_tenure': row.get('franchise_tenure', 0),
            'coach_bellcow_index': row['coach_bellcow_index'],
            'coach_pass_to_rb_share': row['coach_pass_to_rb_share'],
            'coach_te_target_share': row['coach_te_target_share'],
            'first_year_hc_flag': row.get('first_year_hc_flag', 0.0),
        }
    return coach_map

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
    coach_map = compute_coach_metrics(player_stats_df)
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
            
        # Load depth chart for this year
        depth_map = load_depth_charts(year)
            
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
            
            # Get experience, age, height, weight
            years_exp = 0
            age = 26.0  # default median
            height = 73.0
            weight = 215.0
            
            roster_info = roster_map.get(roster_key, roster_map.get((prev_season, clean_pname, pos), None))
            if roster_info:
                years_exp = roster_info['years_exp']
                height = roster_info['height']
                weight = roster_info['weight']
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
                    
            # Get actual points in year Y (for the target variable)
            current_stats_key = (year, clean_pname, pos)
            actual_pts = 0.0
            if current_stats_key in stats_map:
                actual_pts = stats_map[current_stats_key]['fantasy_points']
                    
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
            
            # Depth Chart Features
            pos_rank = depth_map.get((clean_pname, pos), 99)
            is_team_wr1 = 1.0 if (pos == 'WR' and pos_rank == 1) else 0.0
            is_team_wr2 = 1.0 if (pos == 'WR' and pos_rank == 2) else 0.0
            is_team_rb1 = 1.0 if (pos == 'RB' and pos_rank == 1) else 0.0
            
            # Coach History & Age Features
            coach_info = coach_map.get((year, team_ffc), {})
            coach_age = coach_info.get('coach_age', 50.0)
            coach_player_age_gap = coach_age - age
                        
            # Create feature row
            row_dict = {
                'player_name': name,
                'position': pos,
                'year': year,
                'adp': adp_val,
                'actual_pts': actual_pts,
                'is_rookie': is_rookie,
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
                'prev_fantasy_points': prev_stats['fantasy_points'],
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
                'coach_pass_rate': coach_info.get('coach_pass_to_rb_share', 0.5) # Proxy if needed
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
        'is_rookie', 'years_exp', 'age', 'height', 'weight', 'oline_score', 'team_ypc_ex_qb',
        # Pillar 1 Features
        'draft_pick_num', 'draft_round', 'is_day_1', 'is_day_2',
        'is_rookie_top_50', 'rookie_expected_volume',
        'prev_passing_yards', 'prev_passing_tds', 'prev_interceptions',
        'prev_rushing_yards', 'prev_carries', 'prev_rushing_tds',
        'prev_receptions', 'prev_targets', 'prev_receiving_yards',
        'prev_receiving_tds', 'prev_fumbles_lost', 'prev_fantasy_points',
        'is_team_wr1', 'is_team_wr2', 'is_team_rb1',
        'coach_age', 'is_young_innovator', 'is_old_school', 'coach_player_age_gap',
        'coach_experience', 'coach_win_pct', 'franchise_tenure',
        'coach_bellcow_index', 'coach_pass_to_rb_share', 'coach_te_target_share',
        'first_year_hc_flag', 'coach_pass_rate',
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
    y_train_adp = train_data['adp'].values
    y_train_pts = train_data['actual_pts'].values
    
    X_val_raw = val_data[feature_cols].copy()
    y_val_adp = val_data['adp'].values
    y_val_pts = val_data['actual_pts'].values
    
    # Scale features
    scaler = StandardScaler()
    
    # We only fit on train to avoid data leakage
    X_train = scaler.fit_transform(X_train_raw)
    X_val = scaler.transform(X_val_raw)
    
    return X_train, y_train_adp, y_train_pts, X_val, y_val_adp, y_val_pts, scaler, feature_cols

if __name__ == "__main__":
    df = prepare_ml_dataset()
    if len(df) > 0:
        X_train, y_train_adp, y_train_pts, X_val, y_val_adp, y_val_pts, scaler, feature_cols = split_and_scale_data(df)
        print(f"X_train shape: {X_train.shape}, y_train_adp shape: {y_train_adp.shape}, y_train_pts shape: {y_train_pts.shape}")
        print(f"X_val shape: {X_val.shape}, y_val_adp shape: {y_val_adp.shape}, y_val_pts shape: {y_val_pts.shape}")
