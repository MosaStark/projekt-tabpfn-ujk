import numpy as np
import shap
import seaborn as sn
import argparse
import matplotlib.pyplot as plt
import os,os.path
from tqdm import tqdm
import base
import make_results
import dataset
import exp

def make_shap(shap_exp):
    data,splits=shap_exp.get_data()
    clf_type=shap_exp.get_clf()
    def helper(split_i,clf_i):
        train,test=data.divide(split_i)
        if(shap_exp.k is None):
            background_data=train.X
            test_data=test.X
        else:
            background_data = shap.kmeans( train.X, 
                                           shap_exp.k).data
            test_data =shap.kmeans( test.X, 
                                    shap_exp.k).data
        explainer=shap.Explainer( clf_i.model.predict_proba,
                                  background_data)
#        s_test=shap.maskers.Independent(test.X, max_samples=100)
        shap_values = explainer(test_data)#,max_evals=100)
        return shap_values.values
    print(shap_exp.out_path)
    base.make_dir(shap_exp.out_path)
    for i,split_i in enumerate(tqdm(splits)):
        out_i=f"{shap_exp.out_path}/{i}"
        clf_i,_=split_i.fit_clf(data,clf_type())
        print(f"Clf trained:{clf_i}")
        values_i=helper(split_i,clf_i)
        np.savez(out_i, values_i)

def show_shapley( in_path,
                  out_path,
                  id_size=2):
    shap_dirs=[]
    for root, dirs, files in os.walk(in_path):
        for dir_i in dirs:
            path_i=f"{root}/{dir_i}"
            if(is_shap_dir(path_i)):
                shap_dirs.append(path_i)
    base.make_dir(out_path)
    for dir_i in shap_dirs:
        matrix_i=get_matrix(dir_i)
        raw_i=dir_i.split("/")[-id_size:]
        raw_i="_".join(raw_i)
        show_heatmap( matrix_i,
                      raw_i,
                      out_path)

def is_shap_dir(in_path):
    paths=[path_i.split(".")[-1]=="npz" 
             for path_i in base.top_files(in_path)]
    if(len(paths)==0):
        return False
    return all(paths)

def get_matrix(in_path):
    all_shap=[]
    for id_i, path_j in base.iter_files(in_path):
        shap_j=np.load(path_j)["arr_0"]
        all_shap.append(shap_j)
    shap_arr=np.concatenate(all_shap,axis=0)
    return np.mean(shap_arr,axis=0)


def show_heatmap( matrix,
                  title,
                  out_path=None):
    sn.heatmap( matrix,
                cmap="YlGnBu",
                annot=False)#,
    plt.title(title)
    if(out_path):
        out_i=f"{out_path}/{title}"
        plt.tight_layout()
        plt.savefig(out_i,dpi=300, bbox_inches="tight")
        plt.close()
    else:
        plt.show()

def shapley_exp(in_path):
    conf=base.read_json(in_path)
    exp_params=exp.from_dir( conf["data"],
                             conf["split"],
                             conf["clf"])
    out_path=conf["out_path"]
    base.make_dir(out_path)
    for exp_i in exp_params:
        exp_i.out_path=f"{out_path}/{exp_i.id}"
        exp_i.k=conf["k"]
        base.make_dir(exp_i.out_path)
        for exp_j in exp_i.iter_exp("clf_type",conf["clf"]):
            exp_j.out_path+=f"/{exp_j.clf_type}"
            print(exp_j)
            make_shap(exp_j)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--conf_path",type=str,default="conf.json") 
    parser.add_argument("--in_shap",type=str,default="shapley") 
    parser.add_argument("--out_shap",type=str,default="heat") 
    parser.add_argument("--cmd", type=str,default="make")
    args=parser.parse_args()
    if(args.cmd=="make"):
        shapley_exp(args.conf_path)
    if(args.cmd=="show"):
        show_shapley( args.in_shap,
                      args.out_shap)