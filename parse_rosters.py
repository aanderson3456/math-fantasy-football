import json

draft_order = [
    "Primetime Pam", "Chef nico", "Coach Mode Marc", "Hurry-Up Alexander",
    "Trenches Destroyer of Dreams", "Pocket Presence Josh", "Locked In Simon",
    "A-ness's Ice", "Red Zone james", "Ground & Pound Matt"
]

# We will read drafted.txt which has 95 picks.
# We can also append the remaining picks we know.
with open('drafted.txt', 'r') as f:
    lines = [line.strip().replace('[', '').replace(']', '') for line in f.readlines() if line.strip()]

# The user's team is A-ness's Ice (pick 8).
# From the most recent dump, we know some of the picks in rounds 10-15.
# Actually, the user dumped the full draft board up to pick 138!
# Let me just write the draft order parsing logic and we'll see how far we get.

def get_draft_team(pick_num):
    rnd = (pick_num - 1) // 10
    idx = (pick_num - 1) % 10
    if rnd % 2 == 1:
        idx = 9 - idx
    return draft_order[idx]

rosters = {team: [] for team in draft_order}
for i, player in enumerate(lines):
    team = get_draft_team(i + 1)
    rosters[team].append(player)
    
print("Rosters parsed from drafted.txt (first 95 picks):")
for team, players in rosters.items():
    print(f"{team}: {len(players)} players")

with open('rosters.json', 'w') as f:
    json.dump(rosters, f, indent=2)

