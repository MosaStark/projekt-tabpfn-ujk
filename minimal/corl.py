import numpy as np
from dataclasses import dataclass
from collections import defaultdict
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import argparse
import base
import dataset

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
	lines=[]
	plt.rcParams.update({'font.size': 12})
	for key_i,value_i in shap_dict.items():
		r,p=plot(value_i["RF"].as_arr(),
	             value_i["TabPFN"].as_arr(),
	             x_label="RF",
	             y_label="TabPFN",
	             title=key_i)
		lines.append([key_i,r,p])
	df=dataset.make_df(helper=lambda x:x,
                       iterable=lines,
                       cols=["dataset","corl","p"])
	df=df.round(4)
	print(df.to_latex(index=False))

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
        diff_i=df_i["RF"]-df_i["TabPFN"]
        print(id_i)
        print(round(df_i["TabPFN"],4))
        diff.append(diff_i)
        shap_i=shap_dict[id_i]
        x=shap_i["RF"].as_arr()
        y=shap_i["TabPFN"].as_arr()
        corl.append(pearsonr(x, y)[0])
    plot(diff,
	     corl,
	     "diff",
	     "corl",
	     "Shapley values corelation")

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
	return r,p

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("--result", type=str, default="results")
	parser.add_argument("--output", type=str, default="output/matrix")
	parser.add_argument("--cmd", type=str, default="diff")
	args=parser.parse_args()
	if(args.cmd=="diff"):
		diff_corl(args.output,args.result)
	if(args.cmd=="corl"):
		corl_plot(args.output)

