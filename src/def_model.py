import json
import os

# Base prior for any defense (League Average)
PRIOR_DEF_POINTS = 115.0

# 2025 Baseline Likelihood approximations for DEF scoring (Points)
# In a full pipeline this would be scraped from nflverse schedules (Points Allowed, Turnovers)
# We map the 32 teams to their estimated historical likelihood over the last 3 years
HISTORICAL_DEF_LIKELIHOOD = {
    'BAL': 145.0, 'DAL': 140.0, 'SF': 138.0, 'CLE': 135.0, 'NYJ': 132.0,
    'BUF': 130.0, 'KC': 128.0, 'MIA': 125.0, 'NO': 124.0, 'PIT': 122.0,
    'TB': 120.0, 'HOU': 118.0, 'LV': 117.0, 'IND': 115.0, 'JAX': 114.0,
    'CHI': 113.0, 'SEA': 112.0, 'GB': 110.0, 'DEN': 108.0, 'MIN': 105.0,
    'ATL': 104.0, 'CIN': 102.0, 'DET': 100.0, 'NE': 98.0, 'LAR': 95.0,
    'NYG': 92.0, 'TEN': 90.0, 'PHI': 88.0, 'LAC': 85.0, 'CAR': 82.0,
    'ARI': 80.0, 'WAS': 75.0
}

# Modifiers for coaching changes and elite roster movements for 2026
# e.g., Myles Garrett traded to LAR -> huge boost. 
DEF_MODIFIERS = {
    'LAR': 25.0,  # Acquired Myles Garrett, massive pressure increase
    'WAS': 15.0,  # Dan Quinn defense year 2 leap
    'CAR': 10.0,  # Improved roster / defensive coordinator adjustments
    'DAL': -10.0, # Lost defensive coordinator / key pieces
}

def generate_def_projections(year=2026):
    """
    Generate Bayesian DEF projections for the upcoming year.
    Posterior = (Prior * W_prior) + (Likelihood * W_likelihood) + Modifiers
    """
    W_prior = 0.25
    W_likelihood = 0.75
    
    projections = {}
    for team, likelihood in HISTORICAL_DEF_LIKELIHOOD.items():
        # Base bayesian update
        posterior_base = (PRIOR_DEF_POINTS * W_prior) + (likelihood * W_likelihood)
        
        # Apply off-season coach/roster modifiers
        modifier = DEF_MODIFIERS.get(team, 0.0)
        final_projection = posterior_base + modifier
        
        # Add slight gaussian noise in the simulation, but this function returns the expected mean (Points per season)
        # Convert to Points Per Game
        ppg = final_projection / 17.0
        
        projections[f"{team} DEF"] = {
            'points': final_projection,
            'ppg': ppg,
            'vorp': final_projection - 100.0 # roughly baseline replacement DEF
        }
        
    return projections

if __name__ == "__main__":
    proj = generate_def_projections()
    sorted_proj = sorted(proj.items(), key=lambda x: x[1]['points'], reverse=True)
    print("=== 2026 DEF Projections (Bayesian Model) ===")
    for team, data in sorted_proj[:10]:
        print(f"{team}: {data['points']:.1f} pts ({data['ppg']:.1f} PPG)")
