import numpy as np
from dataclasses import dataclass
from collections import defaultdict
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from collections import Counter
import argparse
import base
import pred
import dataset

@dataclass
class ShapleyMatrix:
	data:str
	clf:str
	matrix:np.array

	@property
	def cats(self):
		return self.matrix.shape[1]

	@property
	def feats(self):
		return self.matrix.shape[0]

	def __iter__(self):
		for i in range(self.cats):
			for j in range(self.feats):
				yield (i,j),self.matrix[j][i]
    
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
		arr=[]
		for (i,j),value in self:
			arr.append(value)
		return np.array(arr)
#    	return self.matrix.flatten()

	def __repr__(self):
		return f"{self.data}_{self.clf}"

	def get_cord(self,i):
		clf_i= i % self.cats 
		feat_i= np.ceil(i/self.feats)
		return (int(clf_i),int(feat_i))
    
def residuals(value_i):
    model_i = LinearRegression()
    x=value_i["RF"].as_arr()
    y=value_i["TabPFN"].as_arr()
    x=x.reshape(-1, 1) 
    y=y.reshape(-1, 1) 
    model_i.fit(x, y)
    y_pred = model_i.predict(x)
    res=y-y_pred
    res=(res-np.mean(res))/np.std(res)      	
    return x,res

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

def plot_residuals(matrix_path):
    shap_dict=get_matrices(matrix_path)
    for key_i,value_i in shap_dict.items():
        x,res=residuals(value_i)
        res=np.abs(res)
        plot(x=x.flatten(),
        	 y=res.flatten(),
        	 x_label="RF",
        	 y_label="TabPFN",
        	 title=key_i)

def outliners(matrix_path,
	          result_path="results"):
    shap_dict=get_matrices(matrix_path)
    result_dict=dict(pred.acc_by_clf(result_path))
    x,y=[],[]
    for key_i,value_i in shap_dict.items():
        _,res=residuals(value_i)
        res=res.flatten()
        res=np.abs(res)
        df_i=result_dict[key_i]
        diff_i=df_i["RF"]-df_i["TabPFN"]
        print(key_i)
        x.append(diff_i)
        y.append(np.amax(res))
        res=np.ceil(res)
        count=Counter(res)
        keys=list(count.keys())
        keys.sort()
        print([count[key_i] for key_i in keys])
    plot( x=x,
    	  y=y,
    	  x_label="diff",
    	  y_label="max_residuals",
    	  title="Maximal resuidals")

def outliners_plot(matrix_path,
	               data_path="data"):
    shap_dict=get_matrices(matrix_path)
    size_dict=dataset.cls_sizes(data_path)
    for key_i,value_i in shap_dict.items():
        shap_val=value_i["RF"]
        x,res=residuals(value_i)
        size_i=size_dict[key_i]
        res=res.flatten()
        res=np.abs(res)
        size_vec=[]
        for c in range(shap_val.cats):
        	for f in range(shap_val.feats):
        		size_vec.append(size_i[c])
        res[res<3]=0
        plot(x=size_vec,
        	 y=res,
        	 x_label="class size",
        	 y_label="residuals",
        	 title=key_i)

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("--result", type=str, default="results")
	parser.add_argument("--output", type=str, default="output/matrix")
	parser.add_argument("--cmd", type=str, default="out")
	args=parser.parse_args()
#	raise Exception(args.cmd)
	if(args.cmd=="diff"):
		diff_corl(args.output,args.result)
	if(args.cmd=="corl"):
		corl_plot(args.output)
	if(args.cmd=="res"):
		plot_residuals(args.output)
	if(args.cmd=="out"):
		outliners(args.output)