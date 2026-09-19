import numpy as np
from dataclasses import dataclass
from collections import defaultdict
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import base

@dataclass
class ShapleyMatrix:
    data:str
    clf:str
    matrix:np.array

    @classmethod
    def from_path(cls,in_path):
        id_i=in_path.split("/")[-1]
        id_i=id_i.split(".")[0]
        data_i,clf_i=id_i.split("_")
        matrix_i=np.load(in_path)["arr_0"]
        return cls( data_i,
        	        clf_i,
        	        matrix_i)

    def as_arr(self):
    	return self.matrix.flatten()

    def __repr__(self):
        return f"{self.data}_{self.clf}"

def corl_plot(in_path):
	matrices=[ ShapleyMatrix.from_path(path_i)
	            for path_i in base.top_files(in_path)]
	shap_dict=defaultdict(dict)
	for matrix_i in matrices:
		shap_dict[matrix_i.data][matrix_i.clf]=matrix_i
	plt.rcParams.update({'font.size': 12})
	for key_i,value_i in shap_dict.items():
		x=value_i["RF"].as_arr()
		y=value_i["TabPFN"].as_arr()
		r, p = pearsonr(x, y)
		plt.scatter(x, y, color="steelblue", edgecolor="black", alpha=0.7)
		text=f"\nPearson correlation: r = {r:.4f}, p = {p:.3e}"
		plt.xlabel("RF"+text)
		plt.ylabel("TabPFN")
		plt.title(key_i)
		plt.tight_layout()
		plt.show()

corl_plot("output/matrix")

