import numpy as np
import shap
import seaborn as sn
import argparse
import matplotlib.pyplot as plt
import base
import make_results
import dataset

def make_shap(   data_path, 
                 split_path, 
                 result_path,
                 shapley_path,
                 k=None,
                 clf="RF"):
    base.make_dir(shapley_path)
    clf_type=make_results.CLF_DICT[clf]
    for id_i, path_i in base.iter_files(data_path):
        data_i=dataset.read_csv(path_i)
        shapley_path_i=f"{shapley_path}/{id_i}"
        base.make_dir(shapley_path_i)
        splits_i = base.SplitGroup.read(f"{split_path}/{id_i}")
        for j, split_j in enumerate(splits_i.splits):
            clf_j,_=split_j.fit_clf(data_i,clf_type())
            train,test=data_i.divide(split_j)
            if(k):
                kmeans_summary = shap.kmeans(train.X, k)
                background_data = kmeans_summary.data    
            else:
                background_data = train.X
            explainer=shap.Explainer( clf_j.model.predict_proba, 
                                      background_data)
            shap_values = explainer(test.X).values
            np.savez(f"{shapley_path_i}/{j}", shap_values)

def show_shapley(shapley_path):
    for path_i in base.top_files(shapley_path):
        all_shap=[]
        for id_i, path_j in base.iter_files(path_i):
            shap_j=np.load(path_j)["arr_0"]
            all_shap.append(shap_j)
        shap_arr=np.concatenate(all_shap,axis=0)
        shap_matrix=np.mean(shap_arr,axis=0)
        print(shap_matrix.shape)
        show_heatmap( shap_matrix,
                      path_i)

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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="zbiory_danych")
    parser.add_argument("--split_path", type=str, default="splits")
    parser.add_argument("--result_path", type=str, default="results")
    parser.add_argument("--clf", type=str, default="RF")
    parser.add_argument("--shapley_path", type=str,default="_shapley")
    parser.add_argument("--k", type=str,default=100)
    parser.add_argument("--cmd", type=str,default="show")
    args=parser.parse_args()
    if(args.cmd=="make"):
        make_shap( args.data_path,
               args.split_path,
               args.result_path,
               args.shapley_path,
               args.k,
               args.clf)
    if(args.cmd=="show"):
        show_shapley(args.shapley_path)