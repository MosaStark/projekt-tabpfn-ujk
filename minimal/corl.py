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

class ShapleyGroup:
	def __init__(self,RF,TabPFN):
		self.RF=RF
		self.TabPFN=TabPFN
		self._data=None

	@property
	def data(self):
		if(self._data is None):
			self._data=list(self.RF.keys())
		return self._data

	def __iter__(self):
		for key_i in self.data:
			rf_i=self.RF[key_i]
			tab_i=self.TabPFN[key_i]
			yield key_i,(rf_i,tab_i)

	@classmethod
	def read(cls,in_path):
		shap_dict=defaultdict(dict)
		for path_i in base.top_files(in_path):
			shap_i=ShapleyMatrix.from_path(path_i)
			shap_dict[shap_i.clf][shap_i.data]=shap_i
		return cls( shap_dict["RF"],
    	            shap_dict["TabPFN"])
    
	def resuid(self):
		for id_i,(rf_i,tab_i) in self:
			model_i = LinearRegression()
			x=rf_i.as_arr()
			y=tab_i.as_arr()
			x=x.reshape(-1, 1) 
			y=y.reshape(-1, 1) 
			model_i.fit(x, y)      
			y_pred = model_i.predict(x)
			res=y-y_pred
			res=(res-np.mean(res))/np.std(res)      	
			yield id_i,(x,res)

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

	def __repr__(self):
		return f"{self.data}_{self.clf}"

	def get_cord(self,i):
		clf_i= i % self.cats 
		feat_i= np.ceil(i/self.feats)
		return (int(clf_i),int(feat_i))

def diff_iter(result_path):
    for id_i,df_i in pred.acc_by_clf(result_path):
        diff_i=df_i["RF"]-df_i["TabPFN"]
        yield id_i,diff_i

def diff_corl( matrix_path,
	           result_path):
    shap_dict=ShapleyGroup.read(matrix_path)
    diff_dict=dict(diff_iter(result_path))
    diff,corl=[],[]
    for id_i,(rf_i,tab_i) in shap_dict:
        diff.append(diff_dict[id_i])
        x=rf_i.as_arr()
        y=tab_i.as_arr()
        corl.append(pearsonr(x, y)[0])
    plot(x=diff,
	     y=corl,
	     x_label="diff",
	     y_label="corl",
	     title="Shapley values corelation")

def corl_plot(in_path):
    shap_dict=ShapleyGroup.read(in_path)
    lines=[]
    plt.rcParams.update({'font.size': 12})
    for id_i,(rf_i,tab_i) in shap_dict:
        r,p=plot(rf_i.as_arr(),
                 tab_i.as_arr(),
                 x_label="RF",
                 y_label="TabPFN",
                 title=id_i)
        lines.append([id_i,r,p])
    df=dataset.make_df(helper=lambda x:x,
                       iterable=lines,
                       cols=["dataset","corl","p"])
    df=df.round(4)
    print(df.to_latex(index=False))

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
    shap_dict=ShapleyGroup.read(matrix_path)
    for id_i,(x_i,res_i) in shap_dict.resuid():
        res_i=np.abs(res_i)
        plot(x=x_i.flatten(),
        	 y=res_i.flatten(),
        	 x_label="RF",
        	 y_label="TabPFN",
        	 title=id_i)


def outliners(matrix_path,
	          result_path="results"):
    shap_dict=ShapleyGroup.read(matrix_path)
    diff_dict=dict(diff_iter(result_path))
    diff,max_res=[],[]
    for id_i,(x_i,res_i) in shap_dict.resuid():
        res_i=np.abs(res_i)
        diff.append(diff_dict[id_i])
        max_res.append(np.amax(res_i))
        res_i=np.ceil(res_i)
        res_i=res_i.flatten().tolist()
        count=Counter(res_i)
        keys=list(count.keys())
        keys.sort()
        print([count[key_i] for key_i in keys])
    plot( x=diff,
    	  y=max_res,
    	  x_label="diff",
    	  y_label="max_residuals",
    	  title="Maximal resuidals")

def outliners_plot(matrix_path,
	               data_path="data"):
    shap_dict=ShapleyGroup.read(matrix_path)
    size_dict=dataset.cls_sizes(data_path)
    for id_i,(x_i,res_i) in shap_dict.resuid():
        shap_i=shap_dict.RF[id_i]
        size_i=size_dict[id_i]
        res_i=res_i.flatten()
        res_i=np.abs(res_i)
        size_vec=[]
        for c in range(shap_i.cats):
        	for f in range(shap_i.feats):
        		size_vec.append(size_i[c])
        plot(x=size_vec,
        	 y=res_i,
        	 x_label="class size",
        	 y_label="residuals",
        	 title=id_i)

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("--result", type=str, default="results")
	parser.add_argument("--output", type=str, default="output/matrix")
	parser.add_argument("--cmd", type=str, default="out")
	args=parser.parse_args()
	if(args.cmd=="diff"):
		diff_corl(args.output,args.result)
	if(args.cmd=="corl"):
		corl_plot(args.output)
	if(args.cmd=="res"):
		plot_residuals(args.output)
	if(args.cmd=="out"):
		outliners_plot(args.output)