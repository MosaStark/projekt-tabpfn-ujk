import os
import numpy as np
import shap
import warnings
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from dataset import read_csv
from tabpfn import TabPFNClassifier

warnings.filterwarnings("ignore")

# 1. Parametr 'k' wskazany przez doktora (kompresuje zbiory do reprezentatywnego tła)
K_BACKGROUND = 20  
# Próbka ewaluacyjna (żeby nie blokować procesora na wiele godzin)
EVAL_SAMPLES = 50  

# 2. Lista Twoich 7 zbiorów danych
ZBIORY = [
    "zbiory_danych/cleveland",
    "zbiory_danych/cmc",
     "zbiory_danych/nazwa_zbioru_3", 
     "zbiory_danych/nazwa_zbioru_4",
     "zbiory_danych/nazwa_zbioru_5",
     "zbiory_danych/nazwa_zbioru_6",
     "zbiory_danych/nazwa_zbioru_7"
]

def main():
    # Folder docelowy na wszystkie wygenerowane macierze
    out_dir = "output/shap_matrices"
    os.makedirs(out_dir, exist_ok=True)

    # 3. Definicja modeli
    models = {
        "RF": RandomForestClassifier(random_state=42),
        "GradientBoosting": GradientBoostingClassifier(random_state=42),
        "TabPFN": TabPFNClassifier(device='cuda')
    }

    for data_path in ZBIORY:
        if not os.path.exists(data_path):
            print(f"Pominięto {data_path} - brak pliku/folderu")
            continue
            
        dataset_name = os.path.basename(data_path)
        print(f"\n==========================================")
        print(f"Rozpoczynam analizę zbioru: {dataset_name}")
        print(f"==========================================")
        
        dataset = read_csv(data_path)
        X, y = dataset.X, dataset.y
        
        # Wycinamy mniejszą próbkę do wyliczenia SHAP
        X_eval = X[:EVAL_SAMPLES]
        
        # Kompresja algorytmem K-Means do k punktów
        print(f"Kompresowanie tła algorytmem k-means (k={K_BACKGROUND})...")
        background = shap.kmeans(X, K_BACKGROUND)

        for model_name, clf in models.items():
            print(f"\n>>> Trenowanie modelu {model_name}...")
            clf.fit(X, y)
            
            print(f"Obliczanie macierzy SHAP dla {model_name}...")
            if model_name == "RF":
                explainer = shap.TreeExplainer(clf)
                shap_values = explainer.shap_values(X_eval)
            else:
                explainer = shap.KernelExplainer(clf.predict_proba, background)
                shap_values = explainer.shap_values(X_eval)
            
            # Standaryzacja formatu wyników
            if isinstance(shap_values, list):
                shap_values_matrix = shap_values[1] 
            else:
                shap_values_matrix = shap_values
                
            # Zapis surowej macierzy do pliku .npy
            matrix_path = os.path.join(out_dir, f"shap_{dataset_name}_{model_name}.npy")
            np.save(matrix_path, np.array(shap_values_matrix, dtype=object))
            print(f"Zapisano macierz SHAP: {matrix_path}")

if __name__ == "__main__":
    main()