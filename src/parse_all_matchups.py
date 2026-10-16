import os
import json
import re
from bs4 import BeautifulSoup

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
CACHE_DIR = os.path.join(DATA_DIR, 'cache')

matchup_files = [
    (os.path.join(CACHE_DIR, 'yahoo_matchup_week1.json'), 'Pitts=tits. Shedeur=ass💦💦', 'Burrow my Cook🍆'),
    (os.path.join(CACHE_DIR, 'yahoo_matchup_1.json'), 'Fresh START', 'Kamarageddon'),
    (os.path.join(CACHE_DIR, 'yahoo_matchup_2.json'), 'LaChampion', 'Neugenix Total T'),
    (os.path.join(CACHE_DIR, 'yahoo_matchup_3.json'), 'Kimchi_Stallion', 'Shough Knight'),
    (os.path.join(CACHE_DIR, 'yahoo_matchup_4.json'), '🐼🐨', 'Hawaiian Hammah Squad'),
    (os.path.join(CACHE_DIR, 'yahoo_matchup_5.json'), 'In-Jeanty & Bijan!', 'A$$ Clapper 130')
]

def clean_player_raw(raw):
    m = re.match(r'^(.*?)(?:Video Forecast|Player Note|No new player Notes|Q\b|\b[A-Z][a-z]{2}\s*-\s*[A-Z]+)', raw)
    name = m.group(1).strip() if m else raw.strip()
    name = re.sub(r'[\ue000-\uefff]', '', name).strip()
    
    m_tp = re.search(r'([A-Z][a-z]{0,2}|[A-Z]{2,3})\s*-\s*([A-Z]+)', raw)
    team = m_tp.group(1).upper() if m_tp else 'UNK'
    pos = m_tp.group(2).upper() if m_tp else 'UNK'
    
    m_opp = re.search(r'(Mon|Sun|Thu)\s*[\d:]+\s*(?:am|pm)\s*[@vs]+\s*([A-Z]+)', raw)
    opp = m_opp.group(0) if m_opp else ''
    
    return name, pos, team, opp

all_matchups = []

for idx, (fpath, def_t1, def_t2) in enumerate(matchup_files):
    if not os.path.exists(fpath):
        continue
    with open(fpath, 'r') as f:
        html = json.load(f)
    soup = BeautifulSoup(html, 'html.parser')
    
    t1_proj_total = 0.0
    t2_proj_total = 0.0
    proj_table = soup.find('table', class_='M-a')
    if proj_table:
        rows = proj_table.find_all('tr')
        if len(rows) >= 4:
            tds = [td.get_text(strip=True) for td in rows[3].find_all(['td', 'th'])]
            if len(tds) >= 3:
                try:
                    t1_proj_total = float(tds[0])
                    t2_proj_total = float(tds[2])
                except Exception:
                    pass

    t1_starters = []
    t2_starters = []
    t1_bench = []
    t2_bench = []
    
    # Starters table
    t1_table = soup.find('table', id='statTable1')
    if t1_table:
        for r in t1_table.find_all('tr')[1:]:
            tds = [td.get_text(strip=True) for td in r.find_all('td')]
            if len(tds) >= 10:
                slot = tds[4]
                if slot == 'Total':
                    continue
                p1_raw = tds[1]
                p1_proj = float(tds[2]) if tds[2] and tds[2] != '–' else 0.0
                p2_proj = float(tds[8]) if tds[8] and tds[8] != '–' else 0.0
                p2_raw = tds[9]
                
                p1_name, p1_pos, p1_team, p1_opp = clean_player_raw(p1_raw)
                p2_name, p2_pos, p2_team, p2_opp = clean_player_raw(p2_raw)
                
                t1_starters.append({
                    'slot': slot,
                    'player_name': p1_name,
                    'pos': p1_pos,
                    'team': p1_team,
                    'opp': p1_opp,
                    'yahoo_proj': p1_proj
                })
                t2_starters.append({
                    'slot': slot,
                    'player_name': p2_name,
                    'pos': p2_pos,
                    'team': p2_team,
                    'opp': p2_opp,
                    'yahoo_proj': p2_proj
                })
                
    # Bench table
    t2_table = soup.find('table', id='statTable2')
    if t2_table:
        for r in t2_table.find_all('tr')[1:]:
            tds = [td.get_text(strip=True) for td in r.find_all('td')]
            if len(tds) >= 10:
                slot = tds[4]
                if slot == 'Total':
                    continue
                p1_raw = tds[1]
                p1_proj = float(tds[2]) if tds[2] and tds[2] != '–' else 0.0
                p2_proj = float(tds[8]) if tds[8] and tds[8] != '–' else 0.0
                p2_raw = tds[9]
                
                p1_name, p1_pos, p1_team, p1_opp = clean_player_raw(p1_raw)
                p2_name, p2_pos, p2_team, p2_opp = clean_player_raw(p2_raw)
                
                if p1_name:
                    t1_bench.append({
                        'slot': 'BN',
                        'player_name': p1_name,
                        'pos': p1_pos,
                        'team': p1_team,
                        'opp': p1_opp,
                        'yahoo_proj': p1_proj
                    })
                if p2_name:
                    t2_bench.append({
                        'slot': 'BN',
                        'player_name': p2_name,
                        'pos': p2_pos,
                        'team': p2_team,
                        'opp': p2_opp,
                        'yahoo_proj': p2_proj
                    })

    all_matchups.append({
        'index': idx,
        'team1': {
            'name': def_t1,
            'orig_proj': t1_proj_total,
            'starters': t1_starters,
            'bench': t1_bench
        },
        'team2': {
            'name': def_t2,
            'orig_proj': t2_proj_total,
            'starters': t2_starters,
            'bench': t2_bench
        }
    })

out_path = os.path.join(CACHE_DIR, 'yahoo_week1_all_matchups.json')
with open(out_path, 'w') as f:
    json.dump(all_matchups, f, indent=2)

print(f"Saved {len(all_matchups)} exact Yahoo matchups to {out_path}")
