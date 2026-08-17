import os.path
import argparse
import base, clf, dataset

# --- OSTATECZNA POPRAWKA BŁĘDU WINDOWS (WINERROR 10038) ---
import sys
import select
import time

original_select = select.select
def bezpieczny_select(rlist, wlist, xlist, timeout=None):
    if sys.stdin in rlist:
        rlist = [x for x in rlist if x != sys.stdin]
        if not rlist and not wlist and not xlist:
            time.sleep(timeout or 0)
            return [], [], []
    return original_select(rlist, wlist, xlist, timeout)

select.select = bezpieczny_select
# -----------------------------------------------------------

def make_pred(in_path,out_path):
    clf_types=[clf.RF, clf.GRAD, clf.LR, clf.SVM, clf.TABPFN]
    # ... (reszta kodu bez zmian)
    clf_types=[clf.RF, clf.GRAD, clf.LR, clf.SVM, clf.TABPFN]
    base.make_dir(out_path)
    for id_i,path_i in base.iter_files(in_path):
        print(path_i)
        out_i=f"{out_path}/{id_i}"
        base.make_dir(out_i)
        data_i=dataset.read_csv(path_i)
        splits_i=get_splits(out_i,data_i)
        for type_j in clf_types:
            out_ij=f"{out_i}/{type_j.NAME}"
            base.make_dir(out_ij)
            results,_=splits_i(data_i,type_j)
            results.save(f"{out_ij}/results")
            print(results.get_acc())

def show_pred(in_path):
    reader=base.ResultGroup.read
    for id_i,path_i in base.iter_files(in_path):
        paths=base.filtered_files(path_i,"splits")
        def helper(path_i):
            clf_i=path_i.split("/")[-1]
            result=reader(f"{path_i}/results")
            return [id_i,clf_i,result.get_acc()]
        df=dataset.make_df(helper,
                           iterable=paths,
                           cols=["data","clf","acc"])
        acc=df["acc"].tolist()
        min_acc=min(acc)
        delta_acc= max(acc)-min_acc
        # Zabezpieczenie przed dzieleniem przez zero, gdyby wyniki były identyczne
        if delta_acc == 0:
            delta_acc = 1 
        df["norm_acc"]=df["acc"].apply(lambda acc: (acc-min_acc)/delta_acc)
        print(df)

def get_splits(in_path,data_i):
    split_cls=base.SplitGroup
    split_path=f"{in_path}/splits"
    print(split_path)
    if os.path.exists(split_path):
        return split_cls.read(split_path)
    else:
        splits= split_cls.make( data_i,
                               n_repeats=1,
                               n_splits=10)
        splits.save(split_path)
    return splits

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_path", type=str, default="minimal/zbiory_danych")
    parser.add_argument("--out_path", type=str, default="minimal/output")
    args=parser.parse_args()
    make_pred(args.in_path,args.out_path)
    show_pred(args.out_path)