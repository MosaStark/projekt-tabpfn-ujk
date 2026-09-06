import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.decomposition import PCA
import argparse
import base

class Dataset(object):
    def __init__(self,X,y=None):
        self.X=X
        self.y=y

    def __len__(self):
        return len(self.y)

    def dim(self):
        return self.X.shape[1]
        
    def n_cats(self):
        return int(max(self.y))+1

    def fit_clf(self,train,clf):
        X_train,y_train=self.X[train],self.y[train]
        history=clf.fit(X_train,y_train)
        return clf,history

    def eval(self,train_index,test_index,clf,as_result=True):
        clf,history=self.fit_clf(train_index,clf)
        result=self.pred(test_index,clf)
        return result,history
    
    def pred(self,test_index,clf):
        if(test_index is None):
            X_test,y_test=self.X,self.y
        else:
            X_test,y_test=self.X[test_index],self.y[test_index]
        y_pred=clf.predict(X_test)
        return base.Result(y_pred,y_test)

    def params_dict(self):
        return { "n_cats":self.n_cats(),
                 "dims":(self.dim(),),
                 "n_samples":len(self)}
#                "class_weight":self.weight_dict()}

    def range(self):
        return [ (np.amin(x_i),np.amax(x_i))
                    for x_i in self.X.T]

    def class_sizes(self):
        cats=  list(set(self.y))
        n_cats= len(cats) 
        params={cat_i:0 for cat_i in cats}
        for y_i in self.y:
            params[y_i]+=1
        return params

    def divide(self,split):
        train=self.select(split.train_index)
        test=self.select(split.test_index)
        return train,test
    
    def select(self,index):
        return Dataset(self.X[index],
                       self.y[index])
    def IR(self):
        sizes=self.class_sizes()
        values=list(sizes.values())
        return max(values)/min(values)
    
    def pca_feats(self):
        pca = PCA(n_components=None)
        pca.fit(self.X)
        exp_var=pca.explained_variance_ratio_
        greatest=exp_var[0]
        cum_var=np.cumsum(exp_var)
        int_var=(cum_var>0.95).astype(int)
        thres_var= np.argmax(int_var)/cum_var.shape[0]
        return greatest,thres_var

def read_csv(in_path:str):
    if(type(in_path)==tuple):
        X,y=in_path
        return Dataset(X,y)
    if(type(in_path)!=str):
        return in_path
    df=pd.read_csv(in_path,header=None)
    raw=df.to_numpy()
    X,y=raw[:,:-1],raw[:,-1]
    X= preprocessing.RobustScaler().fit_transform(X)
    return Dataset(X,y)

def make_df(helper,
            iterable,
            cols,
            multi=False):
    lines=[]
    if(multi):
        for arg_i in iterable:
            lines+=helper(arg_i)
    else:
        for arg_i in iterable:
            lines.append(helper(arg_i))
    df=pd.DataFrame.from_records(lines,
                                columns=cols)
    return df 


def make_desc(in_path):
    paths=list(base.iter_files(in_path))
    def helper(args):
        id_i,path_i=args
        data_i=read_csv(path_i)
        line_i=[ id_i,
                 data_i.n_cats(),
                 data_i.dim(),
                 len(data_i),
                 round(data_i.IR(),2)]
        line_i+=data_i.pca_feats()
        return line_i
    cols=["data","classes","feats","samples",
          "IR","pca_max","pca_95"]
    df=make_df(helper,
               iterable=paths,
               cols=cols)
    print(df)
    return df

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_path", type=str,default="test")
    args=parser.parse_args()
    make_desc(args.in_path)