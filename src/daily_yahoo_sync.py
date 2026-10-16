import os
import sys
import json
import re
import argparse
from bs4 import BeautifulSoup

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from src.yahoo_reader import YahooFantasyReader

CACHE_DIR = os.path.join(BASE_DIR, 'data', 'cache')
COOKIE_PATH = os.path.join(BASE_DIR, 'cookie.txt')

LEAGUES = [
    {"id": "772025", "config": "mafia"},
    {"id": "765156", "config": "12_team_full_ppr"}
]

def clean_player_raw(raw):
    m = re.match(r'^(.*?)(?:Video Forecast|Player Note|No new player Notes|New player Note|New\b|Q\b|\b[A-Z][a-z]{2}\s*-\s*[A-Z]+)', raw)
    name = m.group(1).strip() if m else raw.strip()
    name = re.sub(r'[\ue000-\uefff]', '', name).strip()
    name = re.sub(r'(?:New|Player Note|Note)$', '', name).strip()
    name = re.sub(r'(?:IR|PUP|SUSP|NA|Q|O|D)$', '', name).strip()
    name = re.sub(r'\s+', ' ', name).strip()
    
    m_tp = re.search(r'([A-Z][a-z]{0,2}|[A-Z]{2,3})\s*-\s*', raw)
    team = m_tp.group(1).upper() if m_tp else 'UNK'
    
    m_pos = re.search(r'(QB|RB|WR|TE|DEF|K)', raw)
    pos = m_pos.group(1).upper() if m_pos else 'UNK'
    
    status_match = re.search(r'(?:IR|PUP|SUSP|NA|O|D|Q)', raw)
    status = status_match.group(0).upper() if status_match else None
    
    m_opp = re.search(r'(Mon|Sun|Thu)\s*[\d:]+\s*(?:am|pm)\s*[@vs]+\s*([A-Z]+)', raw)
    opp = m_opp.group(0) if m_opp else ''
    
    return name, pos, team, opp, status

def fetch_and_parse_matchup(reader, url, idx):
    html = reader.fetch_url(url)
    if not html:
        return None
        
    soup = BeautifulSoup(html, 'html.parser')
    
    # Get team names
    t1_name, t2_name = f"Team {idx*2+1}", f"Team {idx*2+2}"
    team_headers = soup.select('div.matchup-header h3, section.team h3, a.Tst-team-name, div.F-link.Pend-s a')
    extracted_names = []
    for h in team_headers:
        tname = h.get_text(strip=True)
        if tname and tname not in extracted_names:
            extracted_names.append(tname)
    if len(extracted_names) >= 2:
        t1_name = extracted_names[0]
        t2_name = extracted_names[1]

    t1_actual_total = 0.0
    t2_actual_total = 0.0
    t1_proj_total = 0.0
    t2_proj_total = 0.0
    proj_table = soup.find('table', class_='M-a')
    if proj_table:
        rows = proj_table.find_all('tr')
        if len(rows) >= 3:
            tds_act = [td.get_text(strip=True) for td in rows[2].find_all(['td', 'th'])]
            if len(tds_act) >= 3:
                try:
                    t1_actual_total = float(tds_act[0])
                    t2_actual_total = float(tds_act[2])
                except Exception:
                    pass
        if len(rows) >= 4:
            tds = [td.get_text(strip=True) for td in rows[3].find_all(['td', 'th'])]
            if len(tds) >= 3:
                try:
                    t1_proj_total = float(tds[0])
                    t2_proj_total = float(tds[2])
                except Exception:
                    pass

    t1_starters, t2_starters, t1_bench, t2_bench = [], [], [], []
    
    def parse_table(table_id, is_bench=False):
        table = soup.find('table', id=table_id)
        if not table: return
        for r in table.find_all('tr')[1:]:
            tds = [td.get_text(strip=True) for td in r.find_all('td')]
            if len(tds) >= 10:
                slot = tds[4]
                if slot == 'Total': continue
                p1_raw = tds[1]
                p1_proj = float(tds[2]) if tds[2] and tds[2] != '–' else 0.0
                p1_actual = float(tds[3]) if len(tds) > 3 and tds[3] and tds[3] != '–' else None
                
                p2_actual = float(tds[7]) if len(tds) > 7 and tds[7] and tds[7] != '–' else None
                p2_proj = float(tds[8]) if tds[8] and tds[8] != '–' else 0.0
                p2_raw = tds[9]
                
                p1_name, p1_pos, p1_team, p1_opp, p1_status = clean_player_raw(p1_raw)
                p2_name, p2_pos, p2_team, p2_opp, p2_status = clean_player_raw(p2_raw)
                
                final_slot = 'BN' if is_bench else slot
                if p1_name:
                    p1_entry = {
                        'slot': final_slot, 'player_name': p1_name, 'pos': p1_pos,
                        'team': p1_team, 'opp': p1_opp, 'yahoo_proj': p1_proj, 'status': p1_status
                    }
                    if p1_actual is not None:
                        p1_entry['actual_pts'] = p1_actual
                    (t1_bench if is_bench else t1_starters).append(p1_entry)
                if p2_name:
                    p2_entry = {
                        'slot': final_slot, 'player_name': p2_name, 'pos': p2_pos,
                        'team': p2_team, 'opp': p2_opp, 'yahoo_proj': p2_proj, 'status': p2_status
                    }
                    if p2_actual is not None:
                        p2_entry['actual_pts'] = p2_actual
                    (t2_bench if is_bench else t2_starters).append(p2_entry)

    parse_table('statTable1', False)
    parse_table('statTable2', True)

    return {
        'index': idx,
        'team1': {'name': t1_name, 'orig_proj': t1_proj_total, 'actual_total': t1_actual_total, 'starters': t1_starters, 'bench': t1_bench},
        'team2': {'name': t2_name, 'orig_proj': t2_proj_total, 'actual_total': t2_actual_total, 'starters': t2_starters, 'bench': t2_bench}
    }

def sync_all(target_week=None):
    if not os.path.exists(COOKIE_PATH):
        print(f"Error: {COOKIE_PATH} not found.")
        return False
        
    with open(COOKIE_PATH, 'r') as f:
        cookie_str = f.read().strip()
        
    for l in LEAGUES:
        print(f"\n--- Syncing League {l['id']} ({l['config']}) Week {target_week or 'current'} ---")
        reader = YahooFantasyReader(league_id=l['id'], cookie_string=cookie_str)
        base_url = reader.get_league_base_url()
        if target_week:
            base_url = f"{base_url}?week={target_week}"
        
        # 1. Fetch Scoreboard
        sb_html = reader.fetch_url(base_url)
        if not sb_html:
            print("Failed to fetch league scoreboard page (cookie might be expired)")
            continue
            
        soup = BeautifulSoup(sb_html, 'html.parser')
        links = soup.find_all('a', href=True)
        matchup_links = []
        for a in links:
            href = a['href']
            if ('/matchup?week=' in href or '/matchup?mid1=' in href) and 'mid2=' in href:
                full_url = f"https://football.fantasysports.yahoo.com{href}" if href.startswith('/') else href
                if full_url not in matchup_links:
                    matchup_links.append(full_url)
                    
        print(f"Found {len(matchup_links)} matchup links!")
        
        all_matchups = []
        actuals_dict = {}
        for idx, url in enumerate(matchup_links):
            print(f" Fetching {url}...")
            mdata = fetch_and_parse_matchup(reader, url, idx)
            if mdata:
                all_matchups.append(mdata)
                # Collect actual scores if available
                for side in ['team1', 'team2']:
                    for p in mdata[side]['starters'] + mdata[side]['bench']:
                        if 'actual_pts' in p and p['player_name']:
                            pname = p['player_name']
                            pts = p['actual_pts']
                            actuals_dict[pname] = pts
                            cname = re.sub(r'[\.\',\-]', '', pname.lower()).strip()
                            actuals_dict[cname] = pts
                
        out_name = f"yahoo_week{target_week or 1}_all_matchups_{l['config']}.json"
        out_path = os.path.join(CACHE_DIR, out_name)
        with open(out_path, 'w') as f:
            json.dump(all_matchups, f, indent=2)
        print(f"Saved {len(all_matchups)} matchups to {out_path}")
        
        if l['config'] == 'mafia':
            main_path = os.path.join(CACHE_DIR, "yahoo_week1_all_matchups.json")
            with open(main_path, 'w') as f:
                json.dump(all_matchups, f, indent=2)

        if target_week and actuals_dict and l['config'] == 'mafia':
            actuals_path = os.path.join(CACHE_DIR, f"actuals_week_{target_week}.json")
            with open(actuals_path, 'w') as f:
                json.dump(actuals_dict, f, indent=2)
            print(f"Saved {len(actuals_dict)//2} player actuals to {actuals_path}")

    return True

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Yahoo League Sync")
    parser.add_argument('--week', type=int, default=None, help='Target week to fetch (e.g. 5)')
    args = parser.parse_args()
    sync_all(target_week=args.week)
