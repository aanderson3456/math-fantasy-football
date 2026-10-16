import numpy as np
import random
from numpy.random import normal as npnormal
import sys
import os
import json

# Allow importing from the math-fantasy-football src structure
sys.path.append(os.path.abspath('.'))
os.environ['LEAGUE_CONFIG'] = '10_team_private_half_ppr'
from app import generate_predictions

print("Loading player projections...")
draft_list = generate_predictions()

teams = [
    "Primetime Pam", "Chef nico", "Coach Mode Marc", "Hurry-Up Alexander",
    "Trenches Destroyer of Dreams", "Pocket Presence Josh", "Locked In Simon",
    "A-ness's Ice", "Red Zone james", "Ground & Pound Matt"
]

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

schedule = {}
for week in range(1, 15):
    base_week = ((week - 1) % 9) + 1
    schedule[week] = base_schedule[base_week]

with open('rosters.json', 'r') as f:
    rosters = json.load(f)

draft_list['weekly_mean'] = draft_list['predicted_pts'] / 14.0
draft_list['weekly_std'] = draft_list['weekly_mean'] * 0.4

player_stats = {}
for _, row in draft_list.iterrows():
    player_stats[row['player_name']] = {
        'pos': row['position'],
        'mean': row['weekly_mean'],
        'std': row['weekly_std']
    }

def get_optimal_lineup_projection(roster_names):
    team_players = []
    for name in roster_names:
        if name in player_stats:
            team_players.append(player_stats[name])
        else:
            for p_name, stats in player_stats.items():
                if name.split()[-1] in p_name:
                    team_players.append(stats)
                    break
    
    qbs = sorted([p for p in team_players if p['pos'] == 'QB'], key=lambda x: x['mean'], reverse=True)
    rbs = sorted([p for p in team_players if p['pos'] == 'RB'], key=lambda x: x['mean'], reverse=True)
    wrs = sorted([p for p in team_players if p['pos'] == 'WR'], key=lambda x: x['mean'], reverse=True)
    tes = sorted([p for p in team_players if p['pos'] == 'TE'], key=lambda x: x['mean'], reverse=True)
    defs = sorted([p for p in team_players if p['pos'] == 'DEF'], key=lambda x: x['mean'], reverse=True)

    
    starting_lineup = []
    if qbs: starting_lineup.append(qbs.pop(0))
    if rbs: starting_lineup.append(rbs.pop(0))
    if rbs: starting_lineup.append(rbs.pop(0))
    if wrs: starting_lineup.append(wrs.pop(0))
    if wrs: starting_lineup.append(wrs.pop(0))
    if tes: starting_lineup.append(tes.pop(0))
    if defs: starting_lineup.append(defs.pop(0))
    
    flex_options = rbs + wrs + tes
    flex_options = sorted(flex_options, key=lambda x: x['mean'], reverse=True)
    if flex_options: starting_lineup.append(flex_options.pop(0))
    
    return starting_lineup

def simulate_team_weekly_score(roster_names):
    starting_lineup = get_optimal_lineup_projection(roster_names)
    total_score = 8.0 # Static base for Kicker
    
    for player in starting_lineup:
        score = max(0.0, npnormal(player['mean'], player['std']))
        total_score += score
        
    return total_score

# Optimize team scores to prevent redundant lookup per sim
team_lineups = {}
for team in teams:
    if team in rosters:
        team_lineups[team] = get_optimal_lineup_projection(rosters[team])
    else:
        team_lineups[team] = []

def fast_simulate_team_score(team_name):
    total_score = 15.0
    for player in team_lineups[team_name]:
        score = max(0.0, npnormal(player['mean'], player['std']))
        total_score += score
    return total_score

num_simulations = 2000
playoff_counts = {team: 0 for team in teams}
total_wins = {team: 0 for team in teams}

for _ in range(num_simulations):
    current_standings = {t: [0, 0.0] for t in teams}
    
    for week in range(1, 15):
        for team1, team2 in schedule[week]:
            score1 = fast_simulate_team_score(team1)
            score2 = fast_simulate_team_score(team2)
            
            if score1 > score2:
                current_standings[team1][0] += 1
                current_standings[team1][1] += score1
                current_standings[team2][1] += score2
            else:
                current_standings[team2][0] += 1
                current_standings[team2][1] += score2
                current_standings[team1][1] += score1
                
    sorted_teams = sorted(teams, key=lambda x: (current_standings[x][0], current_standings[x][1]), reverse=True)
    
    for t in sorted_teams[:4]:
        playoff_counts[t] += 1
        
    for t in teams:
        total_wins[t] += current_standings[t][0]
        
print("\\n--- Playoff Probabilities (2026 Preseason) ---")
sorted_results = sorted(playoff_counts.items(), key=lambda item: item[1], reverse=True)

print(f"{'Rank':<5} | {'Team':<30} | {'Avg Wins':<12} | {'Playoff Odds'}")
print("-" * 75)

rank = 1
for team, count in sorted_results:
    percentage = (count / num_simulations) * 100
    avg_w = total_wins[team] / num_simulations
    print(f"{rank:<5} | {team:<30} | {avg_w:<12.1f} | {percentage:.2f}%")
    rank += 1
