import os
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
import base

base_out = "output"
datasets = [d for d in os.listdir(base_out) if os.path.isdir(os.path.join(base_out, d))]

for data_name in datasets:
    data_dir = os.path.join(base_out, data_name)
    results_dir = os.path.join(data_dir, "results")
    if not os.path.exists(results_dir):
        results_dir = data_dir
        
    models = [m for m in os.listdir(results_dir) if os.path.isdir(os.path.join(results_dir, m))]
    
    for model_name in models:
        model_dir = os.path.join(results_dir, model_name)
        try:
            res_group = base.ResultGroup.read(model_dir)
            all_y_true, all_y_pred = [], []
            for r in res_group.results:
                all_y_true.extend(r.y_test)
                all_y_pred.extend(r.y_pred)
            cm = confusion_matrix(all_y_true, all_y_pred)
            out_file = f"matrix_{data_name}_{model_name}.csv"
            pd.DataFrame(cm).to_csv(out_file, index=False, header=False)
            print(f"=== {data_name} | {model_name} ===")
            print(cm)
            print()
        except Exception as e:
            continue