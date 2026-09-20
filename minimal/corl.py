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
	shap_dict=get_matrices(in_path)
	plt.rcParams.update({'font.size': 12})
	for key_i,value_i in shap_dict.items():
		plot(value_i["RF"].as_arr(),
	         value_i["TabPFN"].as_arr(),
	         "RF",
	         "TabPFN",
	         key_i)

def get_matrices(in_path):
	matrices=[ ShapleyMatrix.from_path(path_i)
	            for path_i in base.top_files(in_path)]
	shap_dict=defaultdict(dict)
	for matrix_i in matrices:
		shap_dict[matrix_i.data][matrix_i.clf]=matrix_i
	return shap_dict

def diff_corl( matrix_path,
	           result_path):
    import pred
    shap_dict=get_matrices(matrix_path)
    diff,corl=[],[]
    for id_i,df_i in pred.acc_by_clf(result_path):
        diff.append(df_i["RF"]-df_i["TabPFN"])
        shap_i=shap_dict[id_i]
        x=shap_i["RF"].as_arr()
        y=shap_i["TabPFN"].as_arr()
        corl.append(pearsonr(x, y)[0])
    plot(diff,
	     corl,
	     "diff",
	     "corl",
	     "Corl")

def plot(x,
	     y,
	     x_label,
	     y_label,
	     title):
	r, p = pearsonr(x, y)
	plt.scatter(x, y, color="steelblue", edgecolor="black", alpha=0.7)
	text=f"\nPearson correlation: r = {r:.4f}, p = {p:.3e}"
	plt.xlabel(x_label+text)
	plt.ylabel(y_label)
	plt.title(title)
	plt.tight_layout()
	plt.show()

#corl_plot("output/matrix")
diff_corl("output/matrix","results")

