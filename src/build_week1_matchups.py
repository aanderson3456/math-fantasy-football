import json
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
CACHE_DIR = os.path.join(DATA_DIR, 'cache')

def generate_week1_matchups():
    with open(os.path.join(CACHE_DIR, 'league_10team_player_evals.json'), 'r') as f:
        team_evals = json.load(f)

    # 5 Official Week 1 Matchup pairings from Yahoo:
    matchup_pairs = [
        ("Bijans dont scare me", "Hit Me With Your Prescott"),
        ("Draft Deez Nutz", "MR. SNIFFLES"),
        ("Erica Loves Sports", "Dartin' to the Blue Tent"),
        ("Lizard lizard lizard", "Mahomes a Tear Down"),
        ("Kittle Me This", "Urlach’d n Loaded")
    ]

    matchups_payload = []

    for idx, (t1, t2) in enumerate(matchup_pairs):
        t1_raw_roster = team_evals.get(t1, [])
        t2_raw_roster = team_evals.get(t2, [])

        def build_team_lineup(p_list):
            starters = []
            bench = []
            
            # Find starters: 1 QB, 2 RB, 2 WR, 1 TE, 1 FLEX (RB/WR/TE), 1 K, 1 DEF
            qbs = [p for p in p_list if p['pos'] == 'QB']
            rbs = [p for p in p_list if p['pos'] == 'RB']
            wrs = [p for p in p_list if p['pos'] == 'WR']
            tes = [p for p in p_list if p['pos'] == 'TE']
            ks  = [p for p in p_list if p['pos'] == 'K']
            defs = [p for p in p_list if p['pos'] == 'DEF']

            used_names = set()

            def add_starter(slot, p):
                starters.append({
                    'slot': slot,
                    'player_name': p['player'],
                    'pos': p['pos'],
                    'team': p['nfl'],
                    'yahoo_proj': p['yahoo_w1'],
                    'nn_proj': p['nn_w1']
                })
                used_names.add(p['player'])

            if qbs: add_starter('QB', qbs[0])
            for p in rbs[:2]: add_starter('RB', p)
            for p in wrs[:2]: add_starter('WR', p)
            if tes: add_starter('TE', tes[0])

            # Flex: next best RB or WR or TE
            avail_flex = [p for p in p_list if p['player'] not in used_names and p['pos'] in ['RB', 'WR', 'TE']]
            if avail_flex:
                add_starter('W/R/T', avail_flex[0])

            if ks: add_starter('K', ks[0])
            if defs: add_starter('DEF', defs[0])

            # Bench: everyone else
            for p in p_list:
                if p['player'] not in used_names:
                    bench.append({
                        'slot': 'BN',
                        'player_name': p['player'],
                        'pos': p['pos'],
                        'team': p['nfl'],
                        'yahoo_proj': p['yahoo_w1'],
                        'nn_proj': p['nn_w1']
                    })

            return starters, bench

        t1_starters, t1_bench = build_team_lineup(t1_raw_roster)
        t2_starters, t2_bench = build_team_lineup(t2_raw_roster)

        t1_yahoo_total = round(sum(p['yahoo_proj'] for p in t1_starters), 2)
        t2_yahoo_total = round(sum(p['yahoo_proj'] for p in t2_starters), 2)
        t1_nn_total = round(sum(p['nn_proj'] for p in t1_starters), 2)
        t2_nn_total = round(sum(p['nn_proj'] for p in t2_starters), 2)

        matchups_payload.append({
            "matchup_id": idx + 1,
            "week": 1,
            "team1": {
                "name": t1,
                "yahoo_projected": t1_yahoo_total,
                "nn_projected": t1_nn_total,
                "starters": t1_starters,
                "bench": t1_bench
            },
            "team2": {
                "name": t2,
                "yahoo_projected": t2_yahoo_total,
                "nn_projected": t2_nn_total,
                "starters": t2_starters,
                "bench": t2_bench
            }
        })

    out_file = os.path.join(CACHE_DIR, 'yahoo_week1_all_matchups.json')
    with open(out_file, 'w') as f:
        json.dump(matchups_payload, f, indent=2)

    print(f"Generated {len(matchups_payload)} official Week 1 matchups in {out_file}")

if __name__ == '__main__':
    generate_week1_matchups()
