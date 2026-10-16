import sys

with open('src/preprocessing.py', 'r') as f:
    lines = f.readlines()

new_lines = []
skip = False

# Function to inject
coach_metrics_func = """
def compute_coach_metrics(player_stats_df):
    \"\"\"
    Computes coach history, demographic, and playcalling features.
    Returns: coach_map: {(season, team): coach_features_dict}
    \"\"\"
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
    reg_df = player_stats_df[player_stats_df['season_type'] == 'REG'].copy()
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

"""

for i, line in enumerate(lines):
    if line.startswith("def prepare_ml_dataset():"):
        new_lines.append(coach_metrics_func)
        new_lines.append(line)
    elif "oline_map = compute_oline_scores(player_stats_df)" in line:
        new_lines.append(line)
        new_lines.append("    coach_map = compute_coach_metrics(player_stats_df)\n")
    elif "is_team_rb1 = 1.0 if (pos == 'RB' and pos_rank == 1) else 0.0" in line:
        new_lines.append(line)
        new_lines.append("            \n")
        new_lines.append("            # Coach History & Age Features\n")
        new_lines.append("            coach_info = coach_map.get((year, team_ffc), {})\n")
        new_lines.append("            coach_age = coach_info.get('coach_age', 50.0)\n")
        new_lines.append("            coach_player_age_gap = coach_age - age\n")
    elif "row_dict = {" in line:
        new_lines.append(line)
    elif "'is_team_rb1': is_team_rb1" in line:
        new_lines.append("                'is_team_rb1': is_team_rb1,\n")
        new_lines.append("                # Coach History & Age Features\n")
        new_lines.append("                'coach_age': coach_age,\n")
        new_lines.append("                'is_young_innovator': coach_info.get('is_young_innovator', 0.0),\n")
        new_lines.append("                'is_old_school': coach_info.get('is_old_school', 0.0),\n")
        new_lines.append("                'coach_player_age_gap': coach_player_age_gap,\n")
        new_lines.append("                'coach_experience': coach_info.get('coach_experience', 0),\n")
        new_lines.append("                'coach_win_pct': coach_info.get('coach_win_pct', 0.5),\n")
        new_lines.append("                'franchise_tenure': coach_info.get('franchise_tenure', 0),\n")
        new_lines.append("                'coach_bellcow_index': coach_info.get('coach_bellcow_index', 0.5),\n")
        new_lines.append("                'coach_pass_to_rb_share': coach_info.get('coach_pass_to_rb_share', 0.18),\n")
        new_lines.append("                'coach_te_target_share': coach_info.get('coach_te_target_share', 0.20),\n")
        new_lines.append("                'first_year_hc_flag': coach_info.get('first_year_hc_flag', 0.0),\n")
        new_lines.append("                'coach_pass_rate': coach_info.get('coach_pass_to_rb_share', 0.5) # Proxy if needed\n")
    elif "'is_team_wr1', 'is_team_wr2', 'is_team_rb1'," in line:
        new_lines.append(line)
        new_lines.append("        'coach_age', 'is_young_innovator', 'is_old_school', 'coach_player_age_gap',\n")
        new_lines.append("        'coach_experience', 'coach_win_pct', 'franchise_tenure',\n")
        new_lines.append("        'coach_bellcow_index', 'coach_pass_to_rb_share', 'coach_te_target_share',\n")
        new_lines.append("        'first_year_hc_flag', 'coach_pass_rate',\n")
    else:
        new_lines.append(line)

with open('src/preprocessing.py', 'w') as f:
    f.writelines(new_lines)
