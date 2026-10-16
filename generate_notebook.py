import json
import nbformat as nbf
import os
import sys

nb = nbf.v4.new_notebook()

with open('rosters.json', 'r') as f:
    rosters_json_str = f.read()

cells = []

cells.append(nbf.v4.new_markdown_cell("""## Fantasy Football Playoff Probability Simulation (2026 Edition)
    
This notebook calculates the playoff probabilities for a 10-team fantasy football league using mathematically optimal weekly player projections.

It simulates 10,000 versions of the remaining fantasy season. For each week:
1. It looks up the schedule (or randomizes matchups for weeks without a fixed schedule).
2. It sets the optimal starting lineup for each team based on weekly mean projections.
3. It draws a simulated score for each starting player from a normal distribution.
4. It sums the scores to determine head-to-head winners.
5. It tracks standings and calculates the top 4 playoff teams."""))
    
cells.append(nbf.v4.new_code_cell("""import numpy as np
import random
import copy
from numpy.random import normal as npnormal
import sys
import os
import pandas as pd

# Allow importing from the math-fantasy-football src structure
sys.path.append(os.path.abspath('.'))
os.environ['LEAGUE_CONFIG'] = '10_team_private_half_ppr'
from app import generate_predictions

# Load player predictions
print("Loading player projections...")
draft_list = generate_predictions()
print(f"Loaded {len(draft_list)} player projections.")"""))
    
cells.append(nbf.v4.new_markdown_cell("""### Step 1: Define Teams and Rosters

Here we define the 10 teams and their drafted rosters. Since we have incomplete data for the end of the draft, you can manually add the final few players for each team. The simulation will match the string names here to the names in the `draft_list` DataFrame."""))
    
cells.append(nbf.v4.new_code_cell("""teams = [
    "Primetime Pam", "Chef nico", "Coach Mode Marc", "Hurry-Up Alexander",
    "Trenches Destroyer of Dreams", "Pocket Presence Josh", "Locked In Simon",
    "A-ness's Ice", "Red Zone james", "Ground & Pound Matt"
]

# Provide known schedules
# The format is a dictionary of Week Number -> List of Matchups (Tuples)
# Standard 9-week round-robin repeats starting at Week 10
base_schedule = {
    1: [
        ("A-ness's Ice", "Locked In Simon"),
        ("Trenches Destroyer of Dreams", "Hurry-Up Alexander"),
        ("Primetime Pam", "Red Zone james"), 
        ("Ground & Pound Matt", "Chef nico"),
        ("Coach Mode Marc", "Pocket Presence Josh")
    ],
    2: [
        ("A-ness's Ice", "Coach Mode Marc"),
        ("Trenches Destroyer of Dreams", "Primetime Pam"),
        ("Ground & Pound Matt", "Red Zone james"),
        ("Hurry-Up Alexander", "Chef nico"),
        ("Locked In Simon", "Pocket Presence Josh")
    ],
    3: [
        ("A-ness's Ice", "Pocket Presence Josh"),
        ("Trenches Destroyer of Dreams", "Ground & Pound Matt"),
        ("Hurry-Up Alexander", "Primetime Pam"),
        ("Locked In Simon", "Red Zone james"),
        ("Coach Mode Marc", "Chef nico")
    ],
    4: [
        ("A-ness's Ice", "Chef nico"),
        ("Trenches Destroyer of Dreams", "Locked In Simon"),
        ("Hurry-Up Alexander", "Ground & Pound Matt"),
        ("Primetime Pam", "Pocket Presence Josh"),
        ("Coach Mode Marc", "Red Zone james")
    ],
    5: [
        ("A-ness's Ice", "Red Zone james"),
        ("Trenches Destroyer of Dreams", "Coach Mode Marc"),
        ("Hurry-Up Alexander", "Locked In Simon"),
        ("Primetime Pam", "Ground & Pound Matt"),
        ("Chef nico", "Pocket Presence Josh")
    ],
    6: [
        ("A-ness's Ice", "Trenches Destroyer of Dreams"),
        ("Hurry-Up Alexander", "Coach Mode Marc"),
        ("Primetime Pam", "Locked In Simon"),
        ("Ground & Pound Matt", "Pocket Presence Josh"),
        ("Chef nico", "Red Zone james")
    ],
    7: [
        ("A-ness's Ice", "Hurry-Up Alexander"),
        ("Trenches Destroyer of Dreams", "Chef nico"),
        ("Primetime Pam", "Coach Mode Marc"),
        ("Locked In Simon", "Ground & Pound Matt"),
        ("Pocket Presence Josh", "Red Zone james")
    ],
    8: [
        ("A-ness's Ice", "Primetime Pam"),
        ("Trenches Destroyer of Dreams", "Red Zone james"),
        ("Ground & Pound Matt", "Coach Mode Marc"),
        ("Hurry-Up Alexander", "Pocket Presence Josh"),
        ("Locked In Simon", "Chef nico")
    ],
    9: [
        ("A-ness's Ice", "Ground & Pound Matt"),
        ("Trenches Destroyer of Dreams", "Pocket Presence Josh"),
        ("Hurry-Up Alexander", "Red Zone james"),
        ("Primetime Pam", "Chef nico"),
        ("Locked In Simon", "Coach Mode Marc")
    ]
}

# Expand the 9-week schedule to a full 14-week regular season
schedule = {}
for week in range(1, 15):
    base_week = ((week - 1) % 9) + 1
    schedule[week] = base_schedule[base_week]
"""))

cells.append(nbf.v4.new_code_cell(f"rosters = {rosters_json_str}"))

cells.append(nbf.v4.new_markdown_cell("""### Step 2: Define Simulation Engine

This engine calculates weekly projections per player and simulates normal distribution variance for week-to-week scoring."""))

cells.append(nbf.v4.new_code_cell("""# Add weekly means to our draft_list
# We assume a 14 week regular season.
draft_list['weekly_mean'] = draft_list['predicted_pts'] / 14.0
draft_list['weekly_std'] = draft_list['weekly_mean'] * 0.4 # 40% variance roughly models boom/bust

# Create a fast lookup dict
player_stats = {}
for _, row in draft_list.iterrows():
    # Store the exact name
    player_stats[row['player_name']] = {
        'pos': row['position'],
        'mean': row['weekly_mean'],
        'std': row['weekly_std']
    }

# Function to get optimal lineup projection for a team
def get_optimal_lineup_projection(roster_names):
    team_players = []
    for name in roster_names:
        # Try exact match
        if name in player_stats:
            team_players.append(player_stats[name])
        else:
            # Try fuzzy match based on last name
            for p_name, stats in player_stats.items():
                if name.split()[-1] in p_name:
                    team_players.append(stats)
                    break
                    
    # Requirements: 1 QB, 2 RB, 2 WR, 1 TE, 1 FLEX (RB/WR/TE)
    # Since DEF/K are not projected, we will add a flat baseline of 15 points total for both.
    
    qbs = sorted([p for p in team_players if p['pos'] == 'QB'], key=lambda x: x['mean'], reverse=True)
    rbs = sorted([p for p in team_players if p['pos'] == 'RB'], key=lambda x: x['mean'], reverse=True)
    wrs = sorted([p for p in team_players if p['pos'] == 'WR'], key=lambda x: x['mean'], reverse=True)
    tes = sorted([p for p in team_players if p['pos'] == 'TE'], key=lambda x: x['mean'], reverse=True)
    
    starting_lineup = []
    if qbs: starting_lineup.append(qbs.pop(0))
    if rbs: starting_lineup.append(rbs.pop(0))
    if rbs: starting_lineup.append(rbs.pop(0))
    if wrs: starting_lineup.append(wrs.pop(0))
    if wrs: starting_lineup.append(wrs.pop(0))
    if tes: starting_lineup.append(tes.pop(0))
    
    # Flex
    flex_options = rbs + wrs + tes
    flex_options = sorted(flex_options, key=lambda x: x['mean'], reverse=True)
    if flex_options: starting_lineup.append(flex_options.pop(0))
    
    return starting_lineup

def simulate_team_weekly_score(roster_names):
    starting_lineup = get_optimal_lineup_projection(roster_names)
    total_score = 15.0 # Baseline for K + DEF
    
    for player in starting_lineup:
        # Draw random score from normal distribution
        score = max(0.0, npnormal(player['mean'], player['std']))
        total_score += score
        
    return total_score"""))

cells.append(nbf.v4.new_markdown_cell("""### Step 3: Run the Main Simulation"""))

cells.append(nbf.v4.new_code_cell("""num_simulations = 10000
playoff_counts = {team: 0 for team in teams}
total_wins = {team: 0 for team in teams}

print(f"Running {num_simulations} simulations...")

# To avoid repeating empty lineups for teams not fully entered in `rosters`, 
# we'll give missing teams a default baseline mean score of 100 points
def get_team_score(team_name):
    if team_name in rosters and len(rosters[team_name]) > 5:
        return simulate_team_weekly_score(rosters[team_name])
    else:
        # Default average team if roster isn't fully defined yet
        return max(0.0, npnormal(100.0, 20.0))

for _ in range(num_simulations):
    current_standings = {t: [0, 0.0] for t in teams} # [wins, points]
    
    # Simulate 14 weeks
    for week in range(1, 15):
        if week in schedule and len(schedule[week]) == 5:
            matchups = schedule[week]
        else:
            # Randomize matchups if not in schedule
            shuffled_teams = list(teams)
            random.shuffle(shuffled_teams)
            matchups = [(shuffled_teams[i], shuffled_teams[i+1]) for i in range(0, 10, 2)]
            
        for team1, team2 in matchups:
            score1 = get_team_score(team1)
            score2 = get_team_score(team2)
            
            if score1 > score2:
                current_standings[team1][0] += 1
                current_standings[team1][1] += score1
                current_standings[team2][1] += score2
            else:
                current_standings[team2][0] += 1
                current_standings[team2][1] += score2
                current_standings[team1][1] += score1
                
    # Sort for playoffs (Top 4)
    # Tiebreaker is total points
    sorted_teams = sorted(teams, key=lambda x: (current_standings[x][0], current_standings[x][1]), reverse=True)
    
    for t in sorted_teams[:4]:
        playoff_counts[t] += 1
        
    for t in teams:
        total_wins[t] += current_standings[t][0]
        
print("Simulation complete.")"""))

cells.append(nbf.v4.new_markdown_cell("""### Step 4: Display the Final Results"""))

cells.append(nbf.v4.new_code_cell("""print("\\n--- Playoff Probabilities (2026 Preseason) ---")

sorted_results = sorted(playoff_counts.items(), key=lambda item: item[1], reverse=True)

print(f"{'Rank':<5} | {'Team':<30} | {'Avg Wins':<12} | {'Playoff Odds'}")
print("-" * 75)

rank = 1
for team, count in sorted_results:
    percentage = (count / num_simulations) * 100
    avg_w = total_wins[team] / num_simulations
    print(f"{rank:<5} | {team:<30} | {avg_w:<12.1f} | {percentage:.2f}%")
    rank += 1"""))

nb['cells'] = cells

with open('Fantasy2point0Football2026.ipynb', 'w') as f:
    nbf.write(nb, f)
