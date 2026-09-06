import os
import numpy as np
import shap
import matplotlib.pyplot as plt
import warnings
from dataset import read_csv
from tabpfn import TabPFNClassifier

warnings.filterwarnings("ignore")

def calculate_shapley_tabpfn(data_path):
    print(f"Wczytywanie zbioru: {data_path}")
    dataset = read_csv(data_path)
    X, y = dataset.X, dataset.y

    print("Trenowanie modelu TabPFN...")
    clf = TabPFNClassifier(device='cpu')
    clf.fit(X, y)

    print("Obliczanie wartości Shapleya dla TabPFN (to potrwa dłuższą chwilę!)...")
    
    background = shap.kmeans(X, 10)
    explainer = shap.KernelExplainer(clf.predict_proba, background)
    
    X_sample = X[:50]
    shap_values = explainer.shap_values(X_sample)

    # Upewniamy się, że folder output istnieje
    os.makedirs("output", exist_ok=True)

    matrix_output_path = os.path.join("output", "shap_values_TabPFN.npy")
    np.save(matrix_output_path, np.array(shap_values, dtype=object))
    print(f"Zapisano macierz SHAP jako {matrix_output_path}")

    if isinstance(shap_values, list):
        shap_values_to_plot = shap_values[1] 
    else:
        shap_values_to_plot = shap_values

    print("Generowanie wykresu...")
    shap.summary_plot(shap_values_to_plot, X_sample, show=False)
    
    output_img = "shap_summary_TabPFN.png"
    plt.tight_layout()
    plt.savefig(output_img)
    plt.close()
    print(f"Gotowe! Wykres zapisano jako {output_img}")

if __name__ == "__main__":
    calculate_shapley_tabpfn("zbiory_danych/cleveland")