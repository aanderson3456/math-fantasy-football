import pandas as pd
from app import generate_predictions

if __name__ == "__main__":
    print("Generating predictions...")
    draft_list = generate_predictions()
    
    if len(draft_list) > 0:
        output_path = "data/cache/local_preds.csv"
        draft_list.to_csv(output_path, index=False)
        print(f"Saved predictions to {output_path}")
    else:
        print("Draft list is empty!")
