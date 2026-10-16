import os
import re
import sys
import json
import argparse
import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PREDS_PATH = os.path.join(BASE_DIR, 'data', 'cache', 'live_12team_predictions.parquet')
STATE_PATH = os.path.join(BASE_DIR, 'data', 'cache', 'live_draft_state.json')

LEAGUE_SIZE = 10
MY_SLOT = 7  # 1-indexed (Drafting in Slot 7)

def get_my_picks(rounds=15, num_teams=LEAGUE_SIZE, my_slot=MY_SLOT):
    picks = []
    for r in range(1, rounds + 1):
        if r % 2 == 1:
            p = (r - 1) * num_teams + my_slot
        else:
            p = (r - 1) * num_teams + (num_teams - my_slot + 1)
        picks.append((r, p))
    return picks

MY_PICK_SCHEDULE = get_my_picks()
MY_PICK_NUMBERS = set(p[1] for p in MY_PICK_SCHEDULE)

def load_predictions():
    if os.path.exists(PREDS_PATH):
        return pd.read_parquet(PREDS_PATH)
    csv_path = os.path.join(BASE_DIR, 'data', 'cache', 'local_preds.csv')
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path)
    raise FileNotFoundError("Prediction file not found. Run local_preds.py first.")

def load_state():
    if os.path.exists(STATE_PATH):
        try:
            with open(STATE_PATH, 'r') as f:
                return json.load(f)
        except Exception:
            pass
    return {
        "drafted_players": [],
        "my_roster": []
    }

def save_state(state):
    os.makedirs(os.path.dirname(STATE_PATH), exist_ok=True)
    with open(STATE_PATH, 'w') as f:
        json.dump(state, f, indent=2)

def normalize_name(name):
    name = str(name).lower()
    name = re.sub(r'[\.\',\-]', '', name)
    name = re.sub(r'\b(jr|sr|iii|ii|iv|v)\b', '', name)
    name = re.sub(r'\s+', ' ', name).strip()
    return name

def build_name_lookup(df):
    lookup = {}
    for _, row in df.iterrows():
        pname = row['player_name']
        norm = normalize_name(pname)
        lookup[norm] = pname
        
        # Also build initial + last name: e.g. "j taylor" -> "Jonathan Taylor"
        parts = norm.split()
        if len(parts) >= 2:
            initial_last = f"{parts[0][0]} {parts[-1]}"
            lookup[initial_last] = pname
            
    # Add manual common aliases
    aliases = {
        'j taylor': 'Jonathan Taylor',
        'j gibbs': 'Jahmyr Gibbs',
        'b robinson': 'Bijan Robinson',
        'j chase': "Ja'Marr Chase",
        'p nacua': 'Puka Nacua',
        'c mccaffrey': 'Christian McCaffrey',
        'a st brown': 'Amon-Ra St. Brown',
        'j smith njigba': 'Jaxon Smith-Njigba',
        'j smithnjigba': 'Jaxon Smith-Njigba',
        'j cook': 'James Cook',
        's barkley': 'Saquon Barkley',
        'd henry': 'Derrick Henry',
        'c brown': 'Chase Brown',
        'd achane': "De'Von Achane",
        'k williams': 'Kyren Williams',
        't mcbride': 'Trey McBride',
        'g pickens': 'George Pickens',
        'z flowers': 'Zay Flowers',
        'c olave': 'Chris Olave',
        'n collins': 'Nico Collins',
        'j jacobs': 'Josh Jacobs',
        'b hall': 'Breece Hall',
        't mcmillan': 'Tetairoa McMillan',
        't higgins': 'Tee Higgins',
        'j allen': 'Josh Allen',
        'j hurts': 'Jalen Hurts',
        'l jackson': 'Lamar Jackson',
        'j daniels': 'Jayden Daniels',
        'g wilson': 'Garrett Wilson',
        'c lamb': 'CeeDee Lamb',
        'j jefferson': 'Justin Jefferson',
        'm harrison': 'Marvin Harrison Jr.',
        'm nabers': 'Malik Nabers',
        'b bowers': 'Brock Bowers',
        'j williams': 'Javonte Williams',
        'javonte williams': 'Javonte Williams',
        'k pitts': 'Kyle Pitts',
        'a jeanty': 'Ashton Jeanty',
        'k walker': 'Kenneth Walker III'
    }
    for k, v in aliases.items():
        if normalize_name(k) not in lookup or len(k.split()) == 2:
            lookup[normalize_name(k)] = v
    return lookup

def parse_pasted_text(text, name_lookup, all_players):
    """
    Extracts player names from pasted Yahoo text or plain lists.
    """
    found_players = []
    lines = [l.strip() for l in text.strip().split('\n') if l.strip()]
    
    # Try regex matching on each line
    for line in lines:
        cleaned_line = normalize_name(line)
        # Check direct lookup
        if cleaned_line in name_lookup:
            found_players.append(name_lookup[cleaned_line])
            continue
            
        # Check if line contains player name tokens
        for norm_key, full_name in name_lookup.items():
            if len(norm_key) >= 4 and norm_key in cleaned_line:
                if full_name not in found_players:
                    found_players.append(full_name)
                    
    # Also handle comma-separated or numbered lists: e.g. "11. Justin Jefferson, 12. CeeDee Lamb"
    for chunk in re.split(r'[,;|\n]', text):
        chunk_clean = re.sub(r'^\d+[\.\)\s\-]+', '', chunk.strip())
        norm_chunk = normalize_name(chunk_clean)
        if norm_chunk in name_lookup:
            if name_lookup[norm_chunk] not in found_players:
                found_players.append(name_lookup[norm_chunk])
                
    # Deduplicate while preserving order
    seen = set()
    result = []
    for p in found_players:
        if p not in seen and p in all_players:
            seen.add(p)
            result.append(p)
    return result

def update_draft_state(new_players, raw_input=""):
    state = load_state()
    df = load_predictions()
    all_player_names = set(df['player_name'].values)
    name_lookup = build_name_lookup(df)
    
    if isinstance(new_players, str):
        parsed = parse_pasted_text(new_players, name_lookup, all_player_names)
    else:
        parsed = [name_lookup.get(normalize_name(p), p) for p in new_players if p in all_player_names]
        
    for p in parsed:
        if p not in state['drafted_players']:
            state['drafted_players'].append(p)
            pick_num = len(state['drafted_players'])
            if pick_num in MY_PICK_NUMBERS:
                if p not in state['my_roster']:
                    state['my_roster'].append(p)
                    
    save_state(state)
    return state

def get_strategy_report():
    df = load_predictions()
    state = load_state()
    drafted = set(state.get('drafted_players', []))
    my_roster_names = state.get('my_roster', [])
    
    current_pick = len(drafted) + 1
    current_round = ((current_pick - 1) // LEAGUE_SIZE) + 1
    
    # Find next pick for user
    next_pick = None
    next_round = None
    picks_until = None
    for r, p in MY_PICK_SCHEDULE:
        if p >= current_pick:
            next_pick = p
            next_round = r
            picks_until = p - current_pick
            break
            
    # Filter available
    avail = df[~df['player_name'].isin(drafted)].copy()
    
    # Analyze my roster
    my_roster_df = df[df['player_name'].isin(my_roster_names)]
    pos_counts = {'QB': 0, 'RB': 0, 'WR': 0, 'TE': 0}
    for pos in my_roster_df['position']:
        if pos in pos_counts:
            pos_counts[pos] += 1
            
    # Top available by VORP and predicted points
    avail_sorted = avail.sort_values(by='vorp', ascending=False)
    
    top_wr = avail[avail['position'] == 'WR'].sort_values(by='vorp', ascending=False).head(5)
    top_rb = avail[avail['position'] == 'RB'].sort_values(by='vorp', ascending=False).head(5)
    top_te = avail[avail['position'] == 'TE'].sort_values(by='vorp', ascending=False).head(3)
    top_qb = avail[avail['position'] == 'QB'].sort_values(by='predicted_pts', ascending=False).head(3)
    
    # Determine Contingencies based on roster composition and board
    contingencies = []
    
    # Contingency 1: Best Value / Top Need Fit
    if pos_counts['WR'] == 0 and len(top_wr) > 0:
        c1_player = top_wr.iloc[0]
        contingencies.append({
            "title": f"Primary Target (WR1 Anchor): **{c1_player['player_name']}** ({c1_player['position']} - {c1_player['team_predict']})",
            "reason": f"NN Projects **{c1_player['predicted_pts']:.1f} pts / +{c1_player['vorp']:.1f} VORP**. Fills critical WR1 slot after locking in JT at RB1.",
            "player": c1_player['player_name']
        })
    else:
        c1_player = avail_sorted.iloc[0]
        contingencies.append({
            "title": f"Primary Target (Best Overall Value): **{c1_player['player_name']}** ({c1_player['position']} - {c1_player['team_predict']})",
            "reason": f"NN Projects **{c1_player['predicted_pts']:.1f} pts / +{c1_player['vorp']:.1f} VORP**. Highest value player left on the board.",
            "player": c1_player['player_name']
        })
        
    # Contingency 2: High-Upside Alternative / Pivot
    if len(top_wr) > 1 and pos_counts['WR'] == 0:
        c2_player = top_wr.iloc[1]
        contingencies.append({
            "title": f"Contingency 2 (WR Pivot): **{c2_player['player_name']}** ({c2_player['position']} - {c2_player['team_predict']})",
            "reason": f"Projected **{c2_player['predicted_pts']:.1f} pts / +{c2_player['vorp']:.1f} VORP**. Instant pivot if {contingencies[0]['player']} is taken before pick {next_pick}.",
            "player": c2_player['player_name']
        })
    elif len(top_rb) > 0:
        c2_player = top_rb.iloc[0]
        contingencies.append({
            "title": f"Contingency 2 (Double-RB Hero Stack): **{c2_player['player_name']}** ({c2_player['position']} - {c2_player['team_predict']})",
            "reason": f"Projected **{c2_player['predicted_pts']:.1f} pts / +{c2_player['vorp']:.1f} VORP**. Lock down a devastating dual-bellcow RB core.",
            "player": c2_player['player_name']
        })
        
    # Contingency 3: Elite TE / Positional Advantage or Fallback
    if len(top_te) > 0 and top_te.iloc[0]['vorp'] > 35:
        c3_player = top_te.iloc[0]
        contingencies.append({
            "title": f"Contingency 3 (Elite TE Positional Advantage): **{c3_player['player_name']}** ({c3_player['position']} - {c3_player['team_predict']})",
            "reason": f"Projected **{c3_player['predicted_pts']:.1f} pts / +{c3_player['vorp']:.1f} VORP**. Provides massive weekly positional edge over standard TE tier.",
            "player": c3_player['player_name']
        })
    elif len(top_wr) > 2:
        c3_player = top_wr.iloc[2]
        contingencies.append({
            "title": f"Contingency 3 (WR Fallback): **{c3_player['player_name']}** ({c3_player['position']} - {c3_player['team_predict']})",
            "reason": f"Projected **{c3_player['predicted_pts']:.1f} pts / +{c3_player['vorp']:.1f} VORP**.",
            "player": c3_player['player_name']
        })
        
    return {
        "current_pick": current_pick,
        "current_round": current_round,
        "next_pick": next_pick,
        "next_round": next_round,
        "picks_until": picks_until,
        "my_roster": my_roster_names,
        "pos_counts": pos_counts,
        "contingencies": contingencies,
        "top_wr": top_wr[['player_name', 'team_predict', 'predicted_pts', 'vorp']].to_dict('records'),
        "top_rb": top_rb[['player_name', 'team_predict', 'predicted_pts', 'vorp']].to_dict('records'),
        "top_te": top_te[['player_name', 'team_predict', 'predicted_pts', 'vorp']].to_dict('records'),
        "top_qb": top_qb[['player_name', 'team_predict', 'predicted_pts', 'vorp']].to_dict('records'),
        "drafted_count": len(drafted)
    }

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-i', type=str, help='Raw pasted text or player list')
    parser.add_argument('--reset', action='store_true', help='Reset draft state')
    parser.add_argument('--status', action='store_true', help='Print current status only')
    args = parser.parse_args()
    
    if args.reset:
        save_state({"drafted_players": [], "my_roster": []})
        print("Draft state reset successfully.")
        sys.exit(0)
        
    if args.input:
        update_draft_state(args.input)
        
    rep = get_strategy_report()
    print(json.dumps(rep, indent=2))
