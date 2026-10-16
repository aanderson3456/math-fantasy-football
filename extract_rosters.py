import json

teams = [
    "Primetime Pam", "Chef nico", "Coach Mode Marc", "Hurry-Up Alexander",
    "Trenches Destroyer of Dreams", "Pocket Presence Josh", "Locked In Simon",
    "A-ness's Ice", "Red Zone james", "Ground & Pound Matt"
]

team_aliases = {
    "Trash Panda 6": "Primetime Pam",
    "Pistols at dawn": "Locked In Simon"
}

with open("rosters_dump.txt", "r") as f:
    lines = [line.strip() for line in f.readlines()]
    
rosters = {}
current_team = None

positions = {"QB", "WR", "RB", "TE", "W/R/T", "K", "DEF", "BN"}

i = 0
while i < len(lines):
    line = lines[i]
    
    # Check if this line is a team name
    if any(line.startswith(t) for t in teams + list(team_aliases.keys())):
        team_name = line.strip()
        if team_name in team_aliases:
            team_name = team_aliases[team_name]
        
        # Clean trailing space
        for t in teams:
            if line.startswith(t):
                team_name = t
                
        current_team = team_name
        rosters[current_team] = []
        i += 1
        continue
        
    if current_team and line in positions:
        # The next line might be empty or might be the player name
        if i + 1 < len(lines) and lines[i+1] == "":
            i += 1
        # The next line is the player name
        if i + 1 < len(lines):
            player_name = lines[i+1].strip()
            # DEF often doesn't have a full name, just the team name (e.g. "Seahawks")
            # We skip DEF and K for the simulation's `simulate_team_weekly_score` because it adds a flat 15 points
            # But let's add them anyway
            if player_name:
                rosters[current_team].append(player_name)
            i += 1 # skip the player name line
            # skip the duplicate player name + info line
            i += 1
    
    i += 1
    
# Clean up rosters
import re
# Clean up things like "Q" or "PUP-P" from names
for team in rosters:
    clean_players = []
    for p in rosters[team]:
        # sometimes we just captured the duplicate line because there was no empty space
        pass
    rosters[team] = list(dict.fromkeys(rosters[team])) # unique

with open("rosters.json", "w") as f:
    json.dump(rosters, f, indent=4)
    
print("Rosters written to rosters.json")
