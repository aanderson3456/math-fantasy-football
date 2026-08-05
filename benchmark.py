import torch
import pandas as pd
import numpy as np
from scipy.stats import spearmanr
from src.preprocessing import prepare_ml_dataset, load_aggregate_player_stats
from src.model import FantasyNN, FantasyDataset, train_model
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler

def run_benchmark():
    # 1. Get Base Dataset (features for year Y predict based on Y-1)
    df = prepare_ml_dataset()
    
    # 2. Get Actual Fantasy Points for Year Y
    player_stats_df = load_aggregate_player_stats()
    player_stats_df['clean_name'] = player_stats_df['player_name'].str.lower().str.replace(r'[^a-z0-9]', '', regex=True)
    # We want fantasy points in year Y
    stats_y = player_stats_df[['season', 'clean_name', 'position', 'fantasy_points']].copy()
    stats_y.rename(columns={'season': 'year', 'fantasy_points': 'actual_pts'}, inplace=True)
    
    df['clean_name'] = df['player_name'].str.lower().str.replace(r'[^a-z0-9]', '', regex=True)
    
    # Merge actual points
    df = pd.merge(df, stats_y, on=['year', 'clean_name', 'position'], how='left')
    df['actual_pts'] = df['actual_pts'].fillna(0.0)
    
    # Feature columns
    feature_cols = [
        'is_rookie', 'years_exp', 'age', 'oline_score', 'team_ypc_ex_qb',
        'draft_pick_num', 'draft_round', 'is_day_1', 'is_day_2',
        'is_rookie_top_50', 'rookie_expected_volume',
        'prev_passing_yards', 'prev_passing_tds', 'prev_interceptions',
        'prev_rushing_yards', 'prev_carries', 'prev_rushing_tds',
        'prev_receptions', 'prev_targets', 'prev_receiving_yards',
        'prev_receiving_tds', 'prev_fumbles_lost', 'prev_fantasy_points',
        'rb_oline_score_inter', 'rb_team_ypc_inter',
        'wr_oline_score_inter', 'te_oline_score_inter',
        'pos_QB', 'pos_RB', 'pos_WR', 'pos_TE'
    ]
    
    # One-hot encode position
    df_encoded = pd.get_dummies(df, columns=['position'], prefix='pos')
    for pos in ['QB', 'RB', 'WR', 'TE']:
        col = f'pos_{pos}'
        if col not in df_encoded.columns:
            df_encoded[col] = 0
            
    results = []
    
    for test_year in [2019, 2022, 2023, 2024, 2025]:
        print(f"\n--- Benchmarking Year {test_year} ---")
        
        train_mask = df_encoded['year'] < test_year
        test_mask = df_encoded['year'] == test_year
        
        train_data = df_encoded[train_mask]
        test_data = df_encoded[test_mask]
        
        if len(test_data) == 0:
            print(f"No test data for {test_year}")
            continue
            
        X_train_raw = train_data[feature_cols].copy()
        y_train = train_data['actual_pts'].values # TARGET IS NOW ACTUAL POINTS
        
        X_test_raw = test_data[feature_cols].copy()
        y_test_adp = test_data['adp'].values
        y_test_pts = test_data['actual_pts'].values
        
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train_raw)
        X_test = scaler.transform(X_test_raw)
        
        train_dataset = FantasyDataset(X_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
        val_size = int(0.1 * len(train_dataset))
        train_size = len(train_dataset) - val_size
        train_sub, val_sub = torch.utils.data.random_split(train_dataset, [train_size, val_size])
        
        train_loader_sub = DataLoader(train_sub, batch_size=128, shuffle=True)
        val_loader_sub = DataLoader(val_sub, batch_size=128, shuffle=False)
        
        model = FantasyNN(input_dim=len(feature_cols))
        model, _, _ = train_model(model, train_loader_sub, val_loader_sub, epochs=40, patience=5)
        
        model.eval()
        with torch.no_grad():
            inputs = torch.tensor(X_test, dtype=torch.float32)
            preds = model(inputs).numpy().flatten()
            
        # We are predicting raw points now, so no need to clip to 1.0 (though points should be >= 0)
        preds = np.clip(preds, 0.0, None)
        
        # Calculate Spearman correlation
        # ECR Correlation: How well ADP predicts points
        ecr_corr, _ = spearmanr(y_test_adp, y_test_pts)
        ecr_corr = -ecr_corr # Invert because lower ADP = more points
        
        # Model Correlation: How well Predicted Points predict Actual Points
        model_corr, _ = spearmanr(preds, y_test_pts)
        # We don't invert model_corr because higher predicted points = higher actual points
        
        print(f"Year {test_year} | ECR Spearman: {ecr_corr:.3f} | Model Spearman: {model_corr:.3f}")
        
        results.append({
            'Year': test_year,
            'ECR_Spearman': ecr_corr,
            'Model_Spearman': model_corr,
            'Diff': model_corr - ecr_corr
        })
        
    df_res = pd.DataFrame(results)
    print("\nBenchmark Results:")
    print(df_res)
    df_res.to_csv('benchmark_results.csv', index=False)

if __name__ == "__main__":
    run_benchmark()
