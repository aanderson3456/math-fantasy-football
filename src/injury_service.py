import os
import re
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
CACHE_DIR = os.path.join(DATA_DIR, 'cache')
GAMES_PATH = os.path.join(CACHE_DIR, 'games.parquet')

class InjuryAvailabilityService:
    def __init__(self, season=2026):
        self.season = season
        self.games_df = self._load_games()
        self._all_nfl_teams = set()
        if self.games_df is not None:
            self._all_nfl_teams = set(self.games_df['home_team'].unique()) | set(self.games_df['away_team'].unique())

    def _load_games(self):
        if os.path.exists(GAMES_PATH):
            df = pd.read_parquet(GAMES_PATH)
            return df[df['season'] == self.season]
        return None

    def get_bye_teams_for_week(self, week):
        """
        Determines which NFL teams are on BYE for the specified week using the official schedule.
        """
        if self.games_df is None or len(self.games_df) == 0:
            return set()
        w_games = self.games_df[self.games_df['week'] == week]
        playing = set(w_games['home_team'].unique()) | set(w_games['away_team'].unique())
        return self._all_nfl_teams - playing

    def get_availability_multiplier(self, team, week, status=None):
        """
        Calculates the availability multiplier:
        - 0.0 for Bye weeks, Out, IR, PUP, Suspended
        - 0.25 for Doubtful (historical ~20-25% snap/game rate)
        - 0.85 for Questionable (historical ~75-85% rate)
        - 1.0 for Active / Healthy
        """
        # 1. Bye week check
        if team:
            team_std = str(team).upper().strip()
            bye_teams = self.get_bye_teams_for_week(week)
            if team_std in bye_teams:
                return 0.0

        # 2. Injury status check
        if status:
            s = str(status).upper().strip()
            if any(s.startswith(tag) for tag in ['IR', 'OUT', 'PUP', 'SUSP', 'O']):
                return 0.0
            if 'D' in s: # Doubtful
                return 0.25
            if 'Q' in s: # Questionable
                return 0.85

        return 1.0
