import os
import json
import re
import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
CACHE_DIR = os.path.join(DATA_DIR, 'cache')
PREDS_PATH = os.path.join(CACHE_DIR, 'live_12team_predictions.parquet')
ROSTERS_PATH = os.path.join(DATA_DIR, 'rosters.json')
DRAFT_PICKS_PATH = os.path.join(CACHE_DIR, 'yahoo_full_draft_picks.json')

def normalize_name(name):
    name = str(name).lower()
    name = re.sub(r'[\.\',\-]', '', name)
    name = re.sub(r'\b(jr|sr|iii|ii|iv|v)\b', '', name)
    name = re.sub(r'\s+', ' ', name).strip()
    return name

def grade_from_vorp(vorp, mean_vorp, std_vorp):
    z = (vorp - mean_vorp) / (std_vorp if std_vorp > 0 else 1.0)
    if z >= 1.5: return "A+"
    elif z >= 1.0: return "A"
    elif z >= 0.5: return "A-"
    elif z >= 0.0: return "B+"
    elif z >= -0.5: return "B"
    elif z >= -1.0: return "B-"
    elif z >= -1.5: return "C+"
    elif z >= -2.0: return "C"
    else: return "D"

def analyze_league_draft():
    preds_df = pd.read_parquet(PREDS_PATH)
    
    # Build lookup
    name_to_row = {}
    for _, row in preds_df.iterrows():
        pname = row['player_name']
        norm = normalize_name(pname)
        name_to_row[norm] = row
        # Also initial + last
        parts = norm.split()
        if len(parts) >= 2:
            name_to_row[f"{parts[0][0]} {parts[-1]}"] = row

    # Manual aliases
    aliases = {
        'jonathan taylor': 'Jonathan Taylor',
        'george pickens': 'George Pickens',
        'javonte williams': 'Javonte Williams',
        'courtland sutton': 'Courtland Sutton',
        'kyle pitts': 'Kyle Pitts',
        'kyle pitts sr': 'Kyle Pitts',
        'bo nix': 'Bo Nix',
        'marvin harrison jr': 'Marvin Harrison Jr.',
        'marvin harrison': 'Marvin Harrison Jr.',
        'mark andrews': 'Mark Andrews',
        'tank bigsby': 'Tank Bigsby',
        'brock purdy': 'Brock Purdy',
        'tyjae spears': 'Tyjae Spears',
        'troy franklin': 'Troy Franklin',
        'steelers': 'PIT DEF',
        'steelers def': 'PIT DEF',
        'patriots': 'NE DEF',
        'patriots def': 'NE DEF',
        'rams': 'LAR DEF',
        'broncos': 'DEN DEF',
        'texans': 'HOU DEF',
        'sam laporta': 'Sam LaPorta',
        'dezhaun stribling': 'DeZhaun Stribling',
        'chuba hubbard': 'Chuba Hubbard',
        'blake corum': 'Blake Corum',
        'caleb williams': 'Caleb Williams',
        'juwan johnson': 'Juwan Johnson',
        'matthew stafford': 'Matthew Stafford',
        'aaron jones sr': 'Aaron Jones',
        'dontayvion wicks': 'Dontayvion Wicks',
        'deebo samuel sr': 'Deebo Samuel',
        'tyler allgeier': 'Tyler Allgeier',
        'isiah pacheco': 'Isiah Pacheco',
        'adonai mitchell': 'Adonai Mitchell',
        'kyler murray': 'Kyler Murray',
        'keaton mitchell': 'Keaton Mitchell',
        'isaiah likely': 'Isaiah Likely',
        'myles price': 'Myles Price',
        'wandale robinson': "Wan'Dale Robinson",
        'jadarian price': 'Jeremiah Price',
        'kc concepcion': 'KC Concepcion',
        'michael wilton': 'Michael Wilson'
    }

    with open(DRAFT_PICKS_PATH, 'r') as f:
        draft_picks = json.load(f)

    # Group picks by manager
    manager_picks = {}
    for p in draft_picks:
        m = p['manager']
        if m not in manager_picks:
            manager_picks[m] = []
        manager_picks[m].append(p)

    team_reports = []
    
    for mgr, p_list in manager_picks.items():
        total_pts = 0.0
        total_vorp = 0.0
        pos_pts = {'QB': 0.0, 'RB': 0.0, 'WR': 0.0, 'TE': 0.0, 'DEF': 0.0}
        
        steals = []
        busts = []
        
        matched_roster = []
        for p in p_list:
            pname = p['player_name']
            norm = normalize_name(pname)
            
            row = name_to_row.get(norm)
            if row is None and norm in aliases:
                row = name_to_row.get(normalize_name(aliases[norm]))
            if row is None:
                # Fuzzy fallback
                for k, v in name_to_row.items():
                    if len(k) >= 5 and (k in norm or norm in k):
                        row = v
                        break
                        
            if row is not None:
                pts = float(row['predicted_pts'])
                vorp = float(row['vorp'])
                pos = str(row['position'])
                adp = float(row.get('predicted_adp', 100.0))
            else:
                pts = 100.0
                vorp = 10.0
                pos = p.get('position', 'WR')
                adp = 100.0
                
            pick_num = p['overall_pick']
            # Steal value = adp - pick_num (if model had him at adp 20 and picked at 50 -> +30 steal)
            val_diff = adp - pick_num
            
            if pos in pos_pts:
                pos_pts[pos] += pts
                
            total_pts += pts
            total_vorp += vorp
            
            matched_roster.append({
                'name': pname,
                'pos': pos,
                'pick': pick_num,
                'pts': pts,
                'vorp': vorp,
                'val_diff': val_diff
            })
            
        matched_roster.sort(key=lambda x: x['vorp'], reverse=True)
        # Find best steal & biggest reach
        sorted_by_val = sorted(matched_roster, key=lambda x: x['val_diff'])
        best_steal = sorted_by_val[0] # Lowest predicted ADP vs late pick
        biggest_reach = sorted_by_val[-1] # Highest predicted ADP vs early pick
        
        team_reports.append({
            'manager': mgr,
            'total_pts': total_pts,
            'total_vorp': total_vorp,
            'pos_pts': pos_pts,
            'best_steal': best_steal,
            'biggest_reach': biggest_reach,
            'roster': matched_roster
        })
        
    # Calculate grades
    all_vorps = [t['total_vorp'] for t in team_reports]
    mean_v = np.mean(all_vorps)
    std_v = np.std(all_vorps)
    
    for t in team_reports:
        t['grade'] = grade_from_vorp(t['total_vorp'], mean_v, std_v)
        
    team_reports.sort(key=lambda x: x['total_vorp'], reverse=True)
    for idx, t in enumerate(team_reports, 1):
        t['rank'] = idx
        
    return team_reports

if __name__ == '__main__':
    reps = analyze_league_draft()
    with open(os.path.join(CACHE_DIR, 'draft_grades_report.json'), 'w') as f:
        json.dump(reps, f, indent=2)
    
    print('=== NEURAL NETWORK DRAFT GRADES & VORP RANKINGS ===\n')
    for t in reps:
        print(f"Rank {t['rank']:>2} | Grade: {t['grade']:<2} | VORP: {t['total_vorp']:>6.1f} | Proj Pts: {t['total_pts']:>6.1f} | Team: {t['manager']}")
        print(f"        Best Value Pick: {t['best_steal']['name']} (Pick {t['best_steal']['pick']})")
        print(f"        Biggest Reach:   {t['biggest_reach']['name']} (Pick {t['biggest_reach']['pick']})\n")
