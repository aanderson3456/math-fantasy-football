import json
import re
import pandas as pd
import numpy as np

def run_analysis():
    # 1. Load picks
    with open("data/cache/final_draft_results_10team.json") as f:
        picks = json.load(f)

    # 2. Load Yahoo Week 1 Projections
    with open("/Users/austinanderson/.gemini/antigravity-ide/brain/a74e8d2c-a4d4-4451-99ae-f758bbaef8be/.system_generated/steps/835/output.txt") as f:
        raw_step = f.read()
    start = raw_step.find("{")
    end = raw_step.rfind("}") + 1
    yahoo_w1_raw = json.loads(raw_step[start:end])

    # 3. Load NN Model Predictions
    preds_df = pd.read_parquet("data/cache/live_12team_predictions.parquet")
    comp_df = pd.read_parquet("data/cache/yahoo_vs_nn_draft_comparison.parquet")

    def norm(n):
        n = str(n).lower()
        n = re.sub(r"[\.\',\-\’]", "", n)
        n = re.sub(r"\b(jr|sr|iii|ii|iv|v)\b", "", n)
        return re.sub(r"\s+", " ", n).strip()

    comp_map = {norm(r["player_name"]): r.to_dict() for _, r in comp_df.iterrows()}
    preds_map = {norm(r["player_name"]): r.to_dict() for _, r in preds_df.iterrows()}

    aliases = {
        "jadarian price": "jeremiah price",
        "marshawn lloyd": "marshawn lloyd",
        "dezhaun stribling": "dezhaun stribling",
        "carnell tate": "carnell tate",
        "makai lemon": "makai lemon",
        "mike washington": "mike washington",
        "kyle pitts sr": "kyle pitts",
        "deebo samuel sr": "deebo samuel",
        "aaron jones sr": "aaron jones",
        "chris godwin jr": "chris godwin"
    }

    yahoo_w1_lookup = {}
    for team_id, tdata in yahoo_w1_raw.items():
        for p in tdata.get("players", []):
            pname_clean = norm(p["nameInfo"])
            yahoo_w1_lookup[pname_clean] = p["projPts"]

    team_names = [
        "Draft Deez Nutz", "Erica Loves Sports", "Lizard lizard lizard",
        "Hit Me With Your Prescott", "Mahomes a Tear Down", "Kittle Me This",
        "Dartin' to the Blue Tent", "Urlach’d n Loaded", "Bijans dont scare me",
        "MR. SNIFFLES"
    ]

    team_evals = {t: [] for t in team_names}
    rosters_json_payload = {}

    for p in picks:
        tname = p["fantasyTeam"]
        pname = p["playerName"]
        pos = p["pos"]
        nfl = p["nflTeam"]
        
        n = norm(pname)
        if n in aliases:
            n = aliases[n]
            
        nn_item = comp_map.get(n) or preds_map.get(n) or {}
        
        # NN Proj Season
        nn_season = nn_item.get("nn_proj_pts") or nn_item.get("predicted_pts")
        if nn_season is not None:
            nn_season = round(float(nn_season), 1)
        
        # Yahoo Proj Season
        yahoo_season = nn_item.get("yahoo_proj_pts")
        if yahoo_season is None and pos in ["DEF", "K"]:
            yahoo_season = 115.0 if pos == "DEF" else 135.0
        elif yahoo_season is not None:
            yahoo_season = round(float(yahoo_season), 1)
            
        # Yahoo Week 1
        y_w1 = yahoo_w1_lookup.get(n)
        if y_w1 is None:
            y_w1 = round(yahoo_season / 16.0, 2) if yahoo_season else 0.0
        
        # NN Week 1
        if nn_season:
            if pos == "QB":
                nn_w1 = round((nn_season / 16.0) * 1.55, 2)
            elif pos in ["RB", "WR"]:
                nn_w1 = round((nn_season / 16.0) * 1.05, 2)
            elif pos == "DEF":
                nn_w1 = round((nn_season / 16.0) * 1.25, 2)
            else:
                nn_w1 = round(nn_season / 16.0, 2)
        else:
            nn_w1 = y_w1
            
        diff_season = round(nn_season - yahoo_season, 1) if (nn_season and yahoo_season) else 0.0
        diff_w1 = round(nn_w1 - y_w1, 2)
        
        # Context reasons
        reasons = []
        oline = nn_item.get("oline_score", 0)
        if oline and oline > 0.5:
            reasons.append("Elite OL Protection")
        elif oline and oline < -0.5:
            reasons.append("Downgraded O-Line")
        if nn_item.get("is_young_innovator") == 1.0:
            reasons.append("Modern Scheme Tempo")
        if nn_item.get("coach_bellcow_index", 0.5) > 0.65:
            reasons.append("High Bellcow Touch Volume")
        if nn_item.get("coach_pass_rate", 0.5) > 0.58:
            reasons.append("Pass-Heavy Game Script")
        if nn_item.get("years_exp", 1) <= 2:
            reasons.append("Ascending Young Profile")
            
        item = {
            "player": pname,
            "pos": pos,
            "nfl": nfl,
            "round": p["round"],
            "pick": p["overallPick"],
            "yahoo_season": yahoo_season,
            "nn_season": nn_season,
            "diff_season": diff_season,
            "yahoo_w1": y_w1,
            "nn_w1": nn_w1,
            "diff_w1": diff_w1,
            "reasons": reasons or ["Standard Proj Volume"]
        }
        team_evals[tname].append(item)
        
        if tname not in rosters_json_payload:
            rosters_json_payload[tname] = []
        rosters_json_payload[tname].append({
            "player_name": pname,
            "position": pos,
            "nfl_team": nfl,
            "yahoo_proj": y_w1,
            "nn_proj": nn_w1
        })

    # Save to cache
    with open("data/cache/league_10team_player_evals.json", "w") as f:
        json.dump(team_evals, f, indent=2)

    with open("data/rosters.json", "w") as f:
        json.dump(rosters_json_payload, f, indent=2)

    # Save settings.json
    settings_payload = {
        "league_id": "772025",
        "league_name": "THE Adult Cousin Mafia League",
        "scoring_format": "half_ppr",
        "num_teams": 10,
        "league_teams": team_names,
        "my_team": "Bijans dont scare me",
        "team_id": "9"
    }
    with open("data/settings.json", "w") as f:
        json.dump(settings_payload, f, indent=2)

    print("=== TEAM BY TEAM EVALUATION RESULTS ===")
    for t in team_names:
        players = [p for p in team_evals[t] if p["diff_season"] != 0.0]
        players_sorted = sorted(players, key=lambda x: x["diff_season"], reverse=True)
        underrated = players_sorted[0] if players_sorted else None
        overrated = players_sorted[-1] if players_sorted else None
        print(f"\n==========================================")
        print(f"TEAM: {t}")
        print(f"==========================================")
        if underrated:
            print(f"  [MOST UNDERRATED] {underrated['player']} ({underrated['pos']}-{underrated['nfl']})")
            print(f"    Season: NN {underrated['nn_season']} vs Yahoo {underrated['yahoo_season']} (+{underrated['diff_season']} pts)")
            print(f"    Week 1: NN {underrated['nn_w1']} vs Yahoo {underrated['yahoo_w1']} (+{underrated['diff_w1']} pts)")
            print(f"    Drivers: {', '.join(underrated['reasons'])}")
        if overrated:
            print(f"  [MOST OVERRATED]  {overrated['player']} ({overrated['pos']}-{overrated['nfl']})")
            print(f"    Season: NN {overrated['nn_season']} vs Yahoo {overrated['yahoo_season']} ({overrated['diff_season']} pts)")
            print(f"    Week 1: NN {overrated['nn_w1']} vs Yahoo {overrated['yahoo_w1']} ({overrated['diff_w1']} pts)")
            print(f"    Drivers: {', '.join(overrated['reasons'])}")

if __name__ == '__main__':
    run_analysis()
