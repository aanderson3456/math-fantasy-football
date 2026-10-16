import pandas as pd
from src.config import LEAGUE_CONFIG

class DraftStrategyEngine:
    def __init__(self, all_players, rosters, draft_log, league_teams, my_team=""):
        self.all_players = pd.DataFrame(all_players) if all_players else pd.DataFrame()
        self.rosters = rosters
        self.draft_log = draft_log
        self.my_team = my_team
        
        self.league_teams = league_teams
        self.num_teams = len(league_teams) if league_teams else 10
        
        self.drafted_names = set(p['player_name'] for p in self.draft_log)
        if len(self.all_players) > 0:
            self.available_players = self.all_players[~self.all_players['player_name'].isin(self.drafted_names)].copy()
        else:
            self.available_players = pd.DataFrame()

    def get_current_pick_number(self):
        return len(self.draft_log) + 1

    def get_current_round(self):
        pick = self.get_current_pick_number()
        return ((pick - 1) // self.num_teams) + 1

    def get_team_on_clock(self):
        pick = self.get_current_pick_number()
        current_round = self.get_current_round()
        pick_in_round = (pick - 1) % self.num_teams
        
        # Snake draft order
        if current_round % 2 == 0:
            # Even rounds go backwards
            index = self.num_teams - 1 - pick_in_round
        else:
            index = pick_in_round
            
        if 0 <= index < len(self.league_teams):
            return self.league_teams[index]
        return "Unknown"

    def picks_until_my_turn(self):
        if not self.my_team or self.my_team not in self.league_teams:
            return -1
            
        my_index = self.league_teams.index(self.my_team)
        current_pick = self.get_current_pick_number()
        
        # Look ahead next 30 picks to find my turn
        for offset in range(30):
            pick_check = current_pick + offset
            rnd = ((pick_check - 1) // self.num_teams) + 1
            pick_in_round = (pick_check - 1) % self.num_teams
            
            idx_on_clock = (self.num_teams - 1 - pick_in_round) if rnd % 2 == 0 else pick_in_round
            if idx_on_clock == my_index:
                return offset
                
        return -1

    def get_roster_needs(self):
        if not self.my_team:
            return {}
        my_roster = self.rosters.get(self.my_team, [])
        pos_counts = {'QB': 0, 'RB': 0, 'WR': 0, 'TE': 0}
        for p in my_roster:
            pos = p.get('position')
            if pos in pos_counts:
                pos_counts[pos] += 1
                
        # Simple need heuristics
        needs = []
        if pos_counts['QB'] == 0 and self.get_current_round() > 5:
            needs.append('QB')
        if pos_counts['TE'] == 0 and self.get_current_round() > 5:
            needs.append('TE')
        if pos_counts['RB'] < 4:
            needs.append('RB')
        if pos_counts['WR'] < 4:
            needs.append('WR')
            
        return {'counts': pos_counts, 'needs': needs}

    def detect_tier_drops(self, picks_until_turn):
        if len(self.available_players) == 0:
            return []
            
        warnings = []
        for pos in ['RB', 'WR', 'TE', 'QB']:
            pos_df = self.available_players[self.available_players['position'] == pos].head(6)
            if len(pos_df) < 2:
                continue
                
            pts = pos_df['predicted_pts'].values
            # Check if there is a steep dropoff in projected points within the top available
            if len(pts) >= 3:
                dropoff = pts[0] - pts[-1]
                if dropoff > 15.0: # Arbitrary tier drop threshold
                    # Are there fewer players left in this tier than picks until my turn?
                    # Find where the biggest gap is
                    gaps = [pts[i] - pts[i+1] for i in range(len(pts)-1)]
                    max_gap_idx = gaps.index(max(gaps))
                    if gaps[max_gap_idx] > 8.0:
                        players_in_tier = max_gap_idx + 1
                        if players_in_tier <= (picks_until_turn + 2):
                            warnings.append(f"{pos} tier ending soon! Only {players_in_tier} elite {pos}s left before a major drop-off (-{gaps[max_gap_idx]:.1f} pts).")
        return warnings

    def get_recommendations(self):
        if len(self.available_players) == 0:
            return []
            
        picks_until = self.picks_until_my_turn()
        needs_info = self.get_roster_needs()
        needs = needs_info.get('needs', ['RB', 'WR', 'QB', 'TE'])
        
        recs = []
        
        # 1. Best Overall VORP (The Value Pick)
        best_vorp = self.available_players.sort_values(by='vorp', ascending=False).iloc[0]
        recs.append({
            'player': best_vorp.to_dict(),
            'reason_title': 'Highest VORP',
            'reason': f"Provides the most value over replacement (+{best_vorp['vorp']:.1f})."
        })
        
        # 2. Best Positional Need
        if needs:
            top_need = needs[0]
            need_df = self.available_players[self.available_players['position'].isin(needs)].sort_values(by='vorp', ascending=False)
            if len(need_df) > 0:
                best_need = need_df.iloc[0]
                if best_need['player_name'] != best_vorp['player_name']:
                    recs.append({
                        'player': best_need.to_dict(),
                        'reason_title': 'Positional Need',
                        'reason': f"Fills your roster need for {best_need['position']}."
                    })
                    
        # 3. Best Steal (ADP vs Current Pick)
        current_pick = self.get_current_pick_number()
        self.available_players['steal_value'] = current_pick - self.available_players['predicted_adp']
        steals = self.available_players.sort_values(by='steal_value', ascending=False)
        if len(steals) > 0:
            best_steal = steals.iloc[0]
            if best_steal['steal_value'] > 5.0 and best_steal['player_name'] not in [r['player']['player_name'] for r in recs]:
                recs.append({
                    'player': best_steal.to_dict(),
                    'reason_title': 'Draft Steal',
                    'reason': f"Sliding! Expected ADP was {best_steal['predicted_adp']:.1f} (Pick {current_pick})."
                })

        # Fill to 3 if needed
        idx = 1
        while len(recs) < 3 and idx < len(self.available_players):
            p = self.available_players.sort_values(by='vorp', ascending=False).iloc[idx]
            if p['player_name'] not in [r['player']['player_name'] for r in recs]:
                recs.append({
                    'player': p.to_dict(),
                    'reason_title': 'High Floor / Value',
                    'reason': f"Strong projected points ({p['predicted_pts']:.1f})."
                })
            idx += 1
            
        return recs

    def generate_strategy_payload(self):
        turn = self.picks_until_my_turn()
        team_on_clock = self.get_team_on_clock()
        return {
            'current_pick': self.get_current_pick_number(),
            'current_round': self.get_current_round(),
            'team_on_clock': team_on_clock,
            'is_my_turn': (turn == 0),
            'picks_until_my_turn': turn,
            'roster_needs': self.get_roster_needs(),
            'tier_warnings': self.detect_tier_drops(turn),
            'recommendations': self.get_recommendations()
        }
