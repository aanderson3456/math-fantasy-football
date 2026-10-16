import os
import re
import sys
import json
import argparse
import requests
from bs4 import BeautifulSoup
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
CACHE_DIR = os.path.join(DATA_DIR, 'cache')
ROSTERS_PATH = os.path.join(DATA_DIR, 'rosters.json')
SETTINGS_PATH = os.path.join(DATA_DIR, 'settings.json')
DRAFT_STATE_PATH = os.path.join(CACHE_DIR, 'live_draft_state.json')
MATCHUPS_PATH = os.path.join(DATA_DIR, 'matchups.json')

class YahooFantasyReader:
    def __init__(self, league_id=None, cookie_string=None, cookies_dict=None):
        self.league_id = str(league_id) if league_id else None
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.9',
        })
        if cookie_string:
            self.set_cookies_from_string(cookie_string)
        if cookies_dict:
            for k, v in cookies_dict.items():
                self.session.cookies.set(k, v, domain='.yahoo.com')

    def set_cookies_from_string(self, cookie_string):
        """
        Parses a raw cookie string (e.g. from document.cookie or request headers) and sets on session.
        """
        for item in cookie_string.split(';'):
            if '=' in item:
                k, v = item.strip().split('=', 1)
                self.session.cookies.set(k.strip(), v.strip(), domain='.yahoo.com')

    def get_league_base_url(self):
        if not self.league_id:
            raise ValueError("League ID is required to fetch Yahoo data.")
        return f"https://football.fantasysports.yahoo.com/f1/{self.league_id}"

    def parse_draft_results_html(self, html_content):
        soup = BeautifulSoup(html_content, 'html.parser')
        picks = []
        rows = soup.select('table.Tst-table tbody tr, table#draftresultstable tbody tr, div.draft-results tr')
        if not rows:
            rows = soup.find_all('tr')
            
        for row in rows:
            cols = row.find_all('td')
            if len(cols) < 2:
                continue
                
            text_cells = [c.get_text(strip=True) for c in cols]
            pick_num = None
            player_name = None
            team_name = None
            pos = None
            manager = None
            
            m_pick = re.search(r'^(\d+)\.?', text_cells[0])
            if m_pick:
                pick_num = int(m_pick.group(1))
                
            for cell in cols:
                player_link = cell.find('a', class_=re.compile(r'name|player', re.I))
                if player_link:
                    player_name = player_link.get_text(strip=True)
                    full_txt = cell.get_text(strip=True)
                    m_pos = re.search(r'\b(QB|RB|WR|TE|K|DEF)\b', full_txt, re.I)
                    if m_pos:
                        pos = m_pos.group(1).upper()
                    m_team = re.search(r'([A-Z]{2,3})\s*-\s*(QB|RB|WR|TE|K|DEF)', full_txt)
                    if m_team:
                        team_name = m_team.group(1).upper()
                        
                team_link = cell.find('a', href=re.compile(r'/f1/\d+/\d+', re.I))
                if team_link and not player_link:
                    manager = team_link.get_text(strip=True)
                    
            if player_name:
                picks.append({
                    'pick': pick_num or len(picks) + 1,
                    'player_name': player_name,
                    'nfl_team': team_name or 'UNK',
                    'position': pos or 'UNK',
                    'fantasy_team': manager or (text_cells[-1] if len(text_cells) > 2 else 'Unknown')
                })
        return picks

    def parse_matchup_html(self, html_content):
        """
        Parses Yahoo Fantasy Matchup HTML (/matchup).
        Extracts team names, projected totals, current scores, and roster lineups for both teams.
        """
        soup = BeautifulSoup(html_content, 'html.parser')
        matchup_data = {
            'teams': [],
            'lineups': {}
        }
        
        # Matchup header (Team 1 vs Team 2)
        team_headers = soup.select('div.F-link.Pend-s a, a.Tst-team-name, section.team h3, div.matchup-header h3')
        for th in team_headers:
            tname = th.get_text(strip=True)
            if tname and tname not in matchup_data['teams']:
                matchup_data['teams'].append(tname)
                
        # Parse matchup player tables
        tables = soup.select('table.Tst-table, table.teamtable, table#statTable0, table#statTable1')
        for idx, table in enumerate(tables):
            t_title = matchup_data['teams'][idx] if idx < len(matchup_data['teams']) else f"Team {idx+1}"
            players = []
            for row in table.select('tbody tr'):
                p_link = row.find('a', class_=re.compile(r'name|player', re.I))
                if not p_link:
                    continue
                p_name = p_link.get_text(strip=True)
                row_txt = row.get_text(strip=True)
                m_pos = re.search(r'\b(QB|RB|WR|TE|K|DEF|W/R/T|FLEX|BN|IR)\b', row_txt, re.I)
                slot = m_pos.group(1).upper() if m_pos else 'FLEX'
                
                # Check for projections or points
                pts = 0.0
                pts_match = re.findall(r'\b\d+\.\d+\b', row_txt)
                if pts_match:
                    try:
                        pts = float(pts_match[-1])
                    except Exception:
                        pass
                        
                players.append({
                    'player_name': p_name,
                    'slot': slot,
                    'projected_pts': pts
                })
            if players:
                matchup_data['lineups'][t_title] = players
                
        return matchup_data

    def parse_rosters_html(self, html_content):
        soup = BeautifulSoup(html_content, 'html.parser')
        rosters = {}
        
        team_modules = soup.select('div.team-roster, div.Grid-u-1, section.team, div.teamtable')
        if not team_modules:
            team_modules = [soup]
            
        for mod in team_modules:
            header = mod.find(re.compile(r'h[1-4]|header'), class_=re.compile(r'name|title|team', re.I))
            team_title = header.get_text(strip=True) if header else "My Team"
            
            player_rows = mod.select('table tbody tr')
            player_list = []
            for row in player_rows:
                p_link = row.find('a', class_=re.compile(r'name|player', re.I))
                if not p_link:
                    continue
                p_name = p_link.get_text(strip=True)
                row_text = row.get_text(strip=True)
                m_pos = re.search(r'\b(QB|RB|WR|TE|K|DEF)\b', row_text, re.I)
                pos = m_pos.group(1).upper() if m_pos else 'UNK'
                
                player_list.append({
                    'player_name': p_name,
                    'position': pos
                })
                
            if player_list:
                rosters[team_title] = player_list
        return rosters

    def parse_standings_html(self, html_content):
        soup = BeautifulSoup(html_content, 'html.parser')
        teams = []
        team_links = soup.select('table.Tst-table a.Tst-team-name, div#standingstable a')
        for link in team_links:
            t_name = link.get_text(strip=True)
            if t_name and t_name not in teams:
                teams.append(t_name)
        return teams

    def fetch_url(self, url):
        try:
            resp = self.session.get(url, timeout=15)
            # If redirected to login page, return None
            if "login.yahoo.com" in resp.url:
                print(f"Authentication required: Yahoo redirected {url} to login page.")
                return None
            resp.raise_for_status()
            return resp.text
        except Exception as e:
            print(f"Error fetching {url}: {e}")
            return None

    def sync_from_league(self, my_team_name=None):
        base_url = self.get_league_base_url()
        print(f"Connecting to Yahoo Fantasy League at: {base_url}")
        
        # 1. Draft Results
        draft_url = f"{base_url}/draftresults"
        draft_html = self.fetch_url(draft_url)
        picks = self.parse_draft_results_html(draft_html) if draft_html else []
        
        # 2. Standings & Matchup
        home_html = self.fetch_url(base_url)
        teams = self.parse_standings_html(home_html) if home_html else []
        
        matchup_html = self.fetch_url(f"{base_url}/matchup")
        matchup_data = self.parse_matchup_html(matchup_html) if matchup_html else {}
        
        # Update settings.json
        if teams or matchup_data.get('teams'):
            found_teams = teams or matchup_data.get('teams', [])
            settings = {
                "league_id": self.league_id,
                "scoring_format": "full_ppr" if len(found_teams) >= 12 else "half_ppr",
                "num_teams": len(found_teams) if len(found_teams) > 0 else 12,
                "league_teams": found_teams,
                "my_team": my_team_name or (found_teams[3] if len(found_teams) > 3 else "Your Team")
            }
            with open(SETTINGS_PATH, 'w') as f:
                json.dump(settings, f, indent=2)
            print(f"Synced league settings ({len(found_teams)} teams) to {SETTINGS_PATH}")
            
        if matchup_data.get('lineups'):
            with open(MATCHUPS_PATH, 'w') as f:
                json.dump(matchup_data, f, indent=2)
            print(f"Synced matchup data to {MATCHUPS_PATH}")
            
        return {
            'teams_count': len(teams),
            'picks_count': len(picks),
            'matchup_synced': bool(matchup_data.get('lineups'))
        }

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Yahoo Fantasy Football League Reader & Data Sync")
    parser.add_argument('--league_id', '-l', type=str, help='Yahoo League ID (e.g. 765156)')
    parser.add_argument('--url', '-u', type=str, help='Full Yahoo League URL')
    parser.add_argument('--cookies', '-c', type=str, help='Cookie string from browser')
    parser.add_argument('--my_team', '-m', type=str, help='Your team name')
    parser.add_argument('--file', '-f', type=str, help='Local HTML file or text to parse')
    args = parser.parse_args()
    
    lid = args.league_id
    if not lid and args.url:
        m = re.search(r'/f1/(\d+)', args.url)
        if m:
            lid = m.group(1)
            
    reader = YahooFantasyReader(league_id=lid, cookie_string=args.cookies)
    
    if args.file and os.path.exists(args.file):
        with open(args.file, 'r') as f:
            content = f.read()
        picks = reader.parse_draft_results_html(content)
        rosters = reader.parse_rosters_html(content)
        matchups = reader.parse_matchup_html(content)
        print(f"Parsed from {args.file}: {len(picks)} draft picks, {len(rosters)} rosters, {len(matchups.get('teams', []))} matchup teams")
    elif lid:
        res = reader.sync_from_league(my_team_name=args.my_team)
        print("Sync complete:", res)
    else:
        print("Usage: python src/yahoo_reader.py --league_id <ID> or --url <URL> or --cookies <COOKIE_STRING>")
