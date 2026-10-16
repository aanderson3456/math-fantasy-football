from app import get_or_train_model, generate_predictions
import pandas as pd

draft_list = generate_predictions()
print(draft_list['position'].unique())
