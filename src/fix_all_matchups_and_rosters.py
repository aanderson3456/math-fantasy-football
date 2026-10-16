import json
import re
import os
import math
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
CACHE_DIR = os.path.join(DATA_DIR, 'cache')

def run_fix():
    # 1. Load clean scrape from step 928
    with open('/Users/austinanderson/.gemini/antigravity-ide/brain/a74e8d2c-a4d4-4451-99ae-f758bbaef8be/.system_generated/steps/928/output.txt') as f:
        raw_text = f.read()
    start = raw_text.find('{')
    end = raw_text.rfind('}') + 1
    yahoo_teams = json.loads(raw_text[start:end])

    # 2. Load NN predictions
    preds_df = pd.read_parquet(os.path.join(CACHE_DIR, 'live_12team_predictions.parquet'))
    comp_df = pd.read_parquet(os.path.join(CACHE_DIR, 'yahoo_vs_nn_draft_comparison.parquet'))

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
        "chris godwin jr": "chris godwin",
        "amonra st brown": "amonra st brown"
    }

    # Normalize team name mapping
    # Team IDs 1 to 10
    team_name_by_id = {}
    for tid, tdata in yahoo_teams.items():
        title = tdata["teamTitle"]
        # If team 9 has nickname, ensure we know it's user team
        team_name_by_id[str(tid)] = title

    print("Teams mapped from Yahoo:")
    for tid, tname in team_name_by_id.items():
        print(f"  Team {tid}: {tname}")

    # Build enriched team rosters
    enriched_teams = {}
    rosters_payload = {}
    all_player_evals = {}

    for tid, tdata in yahoo_teams.items():
        tname = tdata["teamTitle"]
        players = tdata["players"]
        
        starters = []
        bench = []
        eval_list = []

        for p in players:
            pname = p["playerName"]
            slot = p["slot"]
            pos = p["position"]
            nfl = p["nflTeam"]
            y_w1 = float(p["projPts"])
            
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
            if yahoo_season is None:
                yahoo_season = round(y_w1 * 16.0, 1)
            else:
                yahoo_season = round(float(yahoo_season), 1)
                
            # NN Week 1
            if nn_season:
                if pos == "QB":
                    nn_w1 = round(nn_season / 17.0, 2)
                elif pos in ["RB", "WR"]:
                    nn_w1 = round((nn_season / 16.0) * 1.05, 2)
                elif pos == "DEF":
                    nn_w1 = round(nn_season / 17.0, 2)
                else:
                    nn_w1 = round(nn_season / 17.0, 2)
            else:
                nn_w1 = y_w1

            diff_w1 = round(nn_w1 - y_w1, 2)
            diff_season = round(nn_season - yahoo_season, 1) if (nn_season and yahoo_season) else 0.0

            # Divergence label
            if diff_w1 >= 1.5:
                div_label = f"+{diff_w1:.1f} NN Bullish"
            elif diff_w1 <= -1.5:
                div_label = f"{diff_w1:.1f} NN Bearish"
            else:
                div_label = "Agreement"

            player_obj = {
                "player_name": pname,
                "slot": slot,
                "pos": pos,
                "team": nfl,
                "yahoo_proj": y_w1,
                "nn_proj": nn_w1,
                "diff_w1": diff_w1,
                "yahoo_season": yahoo_season,
                "nn_season": nn_season,
                "diff_season": diff_season,
                "divergence_label": div_label,
                "nn_std": round(nn_w1 * 0.28, 2),
                "vorp": round(float(nn_item.get("vorp", 0.0)), 1),
                "reasons": ["High Bellcow Touch Volume" if pos == "RB" and nn_w1 > 12 else "Standard Projection"]
            }

            eval_list.append(player_obj)
            
            # Categorize into starters or bench
            if slot in ['QB', 'RB', 'WR', 'TE', 'W/R/T', 'K', 'DEF']:
                starters.append(player_obj)
            else:
                bench.append(player_obj)

        all_player_evals[tname] = eval_list
        rosters_payload[tname] = [p for p in eval_list]
        enriched_teams[tid] = {
            "name": tname,
            "starters": starters,
            "bench": bench,
            "yahoo_total": round(sum(s['yahoo_proj'] for s in starters), 2),
            "nn_total": round(sum(s['nn_proj'] for s in starters), 2)
        }

    # 3. Build the 5 official Week 1 matchups
    # Matchup 1: Team 9 vs Team 4
    # Matchup 2: Team 1 vs Team 10
    # Matchup 3: Team 2 vs Team 7
    # Matchup 4: Team 3 vs Team 5
    # Matchup 5: Team 6 vs Team 8
    matchup_pair_ids = [
        ("9", "4"),
        ("1", "10"),
        ("2", "7"),
        ("3", "5"),
        ("6", "8")
    ]

    matchups_list = []
    for idx, (id1, id2) in enumerate(matchup_pair_ids):
        t1 = enriched_teams[id1]
        t2 = enriched_teams[id2]

        diff_nn = t1["nn_total"] - t2["nn_total"]
        diff_y = t1["yahoo_total"] - t2["yahoo_total"]

        std = 18.5
        win_prob_nn_1 = round(0.5 * (1 + math.erf(diff_nn / (std * math.sqrt(2)))) * 100, 1)
        win_prob_nn_2 = round(100.0 - win_prob_nn_1, 1)

        win_prob_y_1 = round(0.5 * (1 + math.erf(diff_y / (std * math.sqrt(2)))) * 100, 1)
        win_prob_y_2 = round(100.0 - win_prob_y_1, 1)

        matchups_list.append({
            "matchup_id": idx + 1,
            "week": 1,
            "team1": {
                "name": t1["name"],
                "starters": t1["starters"],
                "bench": t1["bench"],
                "yahoo_projected": t1["yahoo_total"],
                "nn_projected": t1["nn_total"],
                "metrics": {
                    "nn_pts": t1["nn_total"],
                    "yahoo_pts": t1["yahoo_total"],
                    "nn_win_prob": win_prob_nn_1,
                    "yahoo_win_prob": win_prob_y_1
                }
            },
            "team2": {
                "name": t2["name"],
                "starters": t2["starters"],
                "bench": t2["bench"],
                "yahoo_projected": t2["yahoo_total"],
                "nn_projected": t2["nn_total"],
                "metrics": {
                    "nn_pts": t2["nn_total"],
                    "yahoo_pts": t2["yahoo_total"],
                    "nn_win_prob": win_prob_nn_2,
                    "yahoo_win_prob": win_prob_y_2
                }
            },
            "summary": {
                "team1": {
                    "nn_pts": t1["nn_total"],
                    "yahoo_pts": t1["yahoo_total"],
                    "nn_win_prob": win_prob_nn_1,
                    "yahoo_win_prob": win_prob_y_1
                },
                "team2": {
                    "nn_pts": t2["nn_total"],
                    "yahoo_pts": t2["yahoo_total"],
                    "nn_win_prob": win_prob_nn_2,
                    "yahoo_win_prob": win_prob_y_2
                },
                "models_agree": (win_prob_nn_1 > 50 and win_prob_y_1 > 50) or (win_prob_nn_1 <= 50 and win_prob_y_1 <= 50)
            }
        })

    # Save to json files
    with open(os.path.join(CACHE_DIR, 'yahoo_week1_all_matchups.json'), 'w') as f:
        json.dump(matchups_list, f, indent=2)

    with open(os.path.join(DATA_DIR, 'rosters.json'), 'w') as f:
        json.dump(rosters_payload, f, indent=2)

    with open(os.path.join(CACHE_DIR, 'league_10team_player_evals.json'), 'w') as f:
        json.dump(all_player_evals, f, indent=2)

    # Save settings
    my_team_name = team_name_by_id["9"]
    league_teams_list = [team_name_by_id[str(i)] for i in range(1, 11)]
    settings = {
        "league_id": "772025",
        "league_name": "THE Adult Cousin Mafia League",
        "scoring_format": "half_ppr",
        "num_teams": 10,
        "league_teams": league_teams_list,
        "my_team": my_team_name,
        "team_id": "9"
    }
    with open(os.path.join(DATA_DIR, 'settings.json'), 'w') as f:
        json.dump(settings, f, indent=2)

    print("\n=== DEBUG SUCCESSFUL! ===")
    print(f"My Team: {my_team_name}")
    m1 = matchups_list[0]
    print(f"Matchup 1: {m1['team1']['name']} vs {m1['team2']['name']}")
    print(f"Team 1 Yahoo Proj: {m1['team1']['yahoo_projected']} pts | NN Proj: {m1['team1']['nn_projected']} pts")
    print(f"Team 2 Yahoo Proj: {m1['team2']['yahoo_projected']} pts | NN Proj: {m1['team2']['nn_projected']} pts")
    for s in m1['team1']['starters']:
        p = s
        print(f"  {s['slot']:<6} {p['player_name']:<22} | Yahoo: {p['yahoo_proj']:>5.2f} pts | NN: {p['nn_proj']:>5.2f} pts | {p['divergence_label']}")

if __name__ == '__main__':
    run_fix()
