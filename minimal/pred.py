import os.path
import argparse
import base, clf, dataset

def show_pred(in_path,verbose=True):
    reader=base.ResultGroup.read
    for id_i,path_i in base.iter_files(in_path):
        paths=base.filtered_files(path_i,"splits")
        def helper(path_i):
            clf_i=path_i.split("/")[-1]
            result=reader(path_i)
            return [id_i,clf_i,result.get_acc()]
        df=dataset.make_df(helper,
                           iterable=paths,
                           cols=["data","clf","acc"])
        acc=df["acc"].tolist()
        min_acc=min(acc)
        delta_acc= max(acc)-min_acc
        if delta_acc == 0:
            delta_acc = 1 
        df["norm_acc"]=df["acc"].apply(lambda acc: (acc-min_acc)/delta_acc)
        if(verbose):
            print(df)
        yield id_i,df

def acc_by_clf(in_path):
    df_iter=show_pred(in_path,verbose=False)
    for id_i,df_i in df_iter:
         yield id_i,df_i.set_index("clf")["norm_acc"]

def fun(in_path):
    print(in_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_path", type=str, default="results")
    args=parser.parse_args()
    list(show_pred(args.result_path))