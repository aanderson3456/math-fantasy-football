import os

# Year range for data collection and training
START_YEAR = 2006
END_YEAR = 2025
PREDICT_YEAR = 2026

# Number of teams in the league (used for VORP calculations)
NUM_TEAMS = 10

# Cache directories for downloaded raw data
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
CACHE_DIR = os.path.join(DATA_DIR, 'cache')

os.makedirs(CACHE_DIR, exist_ok=True)

# Standardized team abbreviation map
# Maps various source abbreviations (nflverse, Fantasy Football Calculator, etc.) to a single standard
TEAM_MAPPING = {
    'ARI': 'ARI', 'ARZ': 'ARI', 'Arizona Cardinals': 'ARI',
    'ATL': 'ATL', 'Atlanta Falcons': 'ATL',
    'BAL': 'BAL', 'BLT': 'BAL', 'Baltimore Ravens': 'BAL',
    'BUF': 'BUF', 'Buffalo Bills': 'BUF',
    'CAR': 'CAR', 'Carolina Panthers': 'CAR',
    'CHI': 'CHI', 'Chicago Bears': 'CHI',
    'CIN': 'CIN', 'Cincinnati Bengals': 'CIN',
    'CLE': 'CLE', 'CLV': 'CLE', 'Cleveland Browns': 'CLE',
    'DAL': 'DAL', 'Dallas Cowboys': 'DAL',
    'DEN': 'DEN', 'Denver Broncos': 'DEN',
    'DET': 'DET', 'Detroit Lions': 'DET',
    'GB': 'GB', 'GNB': 'GB', 'Green Bay Packers': 'GB',
    'HOU': 'HOU', 'Houston Texans': 'HOU',
    'IND': 'IND', 'Indianapolis Colts': 'IND',
    'JAC': 'JAX', 'JAX': 'JAX', 'Jacksonville Jaguars': 'JAX',
    'KC': 'KC', 'KCC': 'KC', 'Kansas City Chiefs': 'KC',
    'LA': 'LAR', 'LAR': 'LAR', 'RAM': 'LAR', 'Los Angeles Rams': 'LAR', 'STL': 'LAR', 'St. Louis Rams': 'LAR',
    'LAC': 'LAC', 'SD': 'LAC', 'SDG': 'LAC', 'Los Angeles Chargers': 'LAC', 'San Diego Chargers': 'LAC',
    'LV': 'LV', 'LVR': 'LV', 'OAK': 'LV', 'Oakland Raiders': 'LV', 'Las Vegas Raiders': 'LV',
    'MIA': 'MIA', 'Miami Dolphins': 'MIA',
    'MIN': 'MIN', 'Minnesota Vikings': 'MIN',
    'NE': 'NE', 'NWE': 'NE', 'New England Patriots': 'NE',
    'NO': 'NO', 'NOR': 'NO', 'New Orleans Saints': 'NO',
    'NYG': 'NYG', 'New York Giants': 'NYG',
    'NYJ': 'NYJ', 'New York Jets': 'NYJ',
    'PHI': 'PHI', 'Philadelphia Eagles': 'PHI',
    'PIT': 'PIT', 'Pittsburgh Steelers': 'PIT',
    'SF': 'SF', 'SFO': 'SF', 'San Francisco 49ers': 'SF',
    'SEA': 'SEA', 'Seattle Seahawks': 'SEA',
    'TB': 'TB', 'TAM': 'TB', 'Tampa Bay Buccaneers': 'TB',
    'TEN': 'TEN', 'Tennessee Titans': 'TEN',
    'WAS': 'WAS', 'WSH': 'WAS', 'Washington Redskins': 'WAS', 'Washington Football Team': 'WAS', 'Washington Commanders': 'WAS'
}

def standardize_team(team_str):
    if not team_str:
        return 'UNK'
    team_str = str(team_str).strip()
    # Remove city names or full franchise suffixes if direct lookup fails
    standardized = TEAM_MAPPING.get(team_str, None)
    if standardized:
        return standardized
    standardized = TEAM_MAPPING.get(team_str.upper(), None)
    if standardized:
        return standardized
    return team_str.upper()[:3]

# Manual team O-line quality adjustments for PREDICT_YEAR
# Keys are team abbreviations (standardized), values are adjustment offsets to oline_score
TEAM_OLINE_ADJUSTMENTS = {
    'SEA': 1.0,   # Boost Seattle Seahawks O-line & priority
    'MIA': -1.0,  # Miami sucks
    'DET': -0.5,  # Downgrade Detroit's O-line
}

