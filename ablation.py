import os
import numpy as np
from dataset import Dataset
from clf import get_clf

def run_ablation(data_path, output_path):
    # Wczytujemy zbiór danych
    dataset = Dataset.read(data_path)
    X, y = dataset.X, dataset.y
    n_features = X.shape[1]
    
    print(f"Rozpoczynam ablację cech dla zbioru: {data_path}")
    print(f"Liczba cech w zbiorze: {n_features}")

    classifiers = ['RF', 'TabPFN'] 
    
    results = {}

    for clf_name in classifiers:
        print(f"\nTestowanie klasyfikatora: {clf_name}")
        clf = get_clf(clf_name)
        
       
        
        feature_importances = []
        
        for feature_idx in range(n_features):
            X_ablated = X.copy()
            X_ablated[:, feature_idx] = 0 # Lub np. losowa permutacja kolumny
            
            
            
        results[clf_name] = feature_importances

    return results

if __name__ == "__main__":
    data_dir = "minimal/zbiory_danych/cleveland"
    run_ablation(data_dir, "minimal/output/ablation_results")