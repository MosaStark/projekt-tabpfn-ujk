import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import RepeatedStratifiedKFold
import os

class SplitGroup(object):
    def __init__(self,splits):
        self.splits=splits

    def __len__(self):
        return len(self.splits)

    def __call__(self,data,clf_type):
        all_results,all_clfs=[],[]
        for split_i in tqdm(self.splits):
            clf_i=clf_type()
            split_i.fit_clf(data,clf_i)
            result_i=split_i.pred(data,clf_i)
            all_results.append(result_i)
            all_clfs.append(clf_i)
        return ResultGroup(all_results),all_clfs

    @classmethod
    def make( cls,
              data,
              n_repeats=1,
              n_splits=10):
        rskf=RepeatedStratifiedKFold(n_repeats=n_repeats, 
                                     n_splits=n_splits, 
                                     random_state=0)
        splits=[]
        for train_index,test_index in rskf.split(data.X,data.y):
            splits.append(Split(train_index,test_index))
        return cls(splits)
    
    def save(self,out_path):
        make_dir(out_path)
        for i,split_i in enumerate(self.splits):
            split_i.save(f"{out_path}/{i}")
    
    @classmethod
    def read(cls,in_path):
        splits=[ Split.read(path_i)
                  for path_i in top_files(in_path)]
        return cls(splits)

class Split(object):
    def __init__(self,train_index,test_index):
        self.train_index=train_index
        self.test_index=test_index
    
    def fit_clf(self,data,clf):
        return data.fit_clf(self.train_index,clf)

    def pred(self,data,clf):
        return data.pred(self.test_index,
                         clf=clf)

    def save(self,out_path):
        return np.savez(out_path,self.train_index,self.test_index)

    @classmethod
    def read(cls,in_path):
        raw=list(np.load(in_path).values())
        train_index,test_index=raw[0],raw[1]
        return cls( train_index=train_index,
                    test_index=test_index)

    def __str__(self):
        train_size=self.train_index.shape[0]
        test_size=self.test_index.shape[0]
        return f"train:{train_size},test:{test_size}"    
    
    @classmethod
    def random(cls,n_samples,p=0.9):
        # ... (pominięto Dataset ze względu na brak definicji, zachowano Twoją logikę)
        train,test=[],[]
        for i in range(n_samples):
            if(np.random.rand()<p):
                train.append(i)
            else:
                test.append(i)
        train,test=np.array(train),np.array(test)
        return cls( train_index=train,
                    test_index=test)

class Result(object):
    def __init__(self,y_pred,y_true):
        self.y_pred=y_pred
        self.y_true=y_true

    def get_acc(self):
        # Dla accuracy_score kolejność nie ma znaczenia, ale dobrą praktyką jest (y_true, y_pred)
        return accuracy_score(self.y_true, self.y_pred)
    
    def get_f1(self):
        # Oblicza f1-score osobno dla każdej klasy, co jest wymagane w ewaluacji
        return f1_score(self.y_true, self.y_pred, average=None)
  
    def save(self,out_path):
        y_pair=np.array([self.y_pred,self.y_true])
        np.savez(out_path,y_pair)
    
    @classmethod
    def read(cls,in_path:str):
        if(type(in_path)==Result):
            return in_path
        raw=list(np.load(in_path).values())[0]
        y_pred,y_true=raw[0],raw[1]
        return cls( y_pred=y_pred,
                    y_true=y_true)
    
class ResultGroup(object):
    def __init__(self,indiv_result):
        self.indiv_result=indiv_result

    def get_acc(self):
        acc=[ result_i.get_acc() 
                for result_i in self.indiv_result]
        return np.mean(acc)

    def get_f1(self):
        # Wyciąga f1-score z poszczególnych podziałów i uśrednia wyniki dla każdej klasy
        f1_scores = [result_i.get_f1() for result_i in self.indiv_result]
        return np.mean(f1_scores, axis=0)

    def save(self,out_path):
        make_dir(out_path)
        for i,result_i in enumerate(self.indiv_result):
            result_i.save(f"{out_path}/{i}")
    
    @classmethod
    def read(cls,in_path:str):
        results=[ Result.read(path_i)
                  for path_i in top_files(in_path)]
        return cls(results)

def make_dir(path):
    if(not os.path.isdir(path)):
        os.mkdir(path)

def top_files(path):
    if(type(path)==str):
        paths=[ f'{path}/{file_i}' for file_i in os.listdir(path)]
    else:
        paths=path
    return paths

def iter_files(path):
    for file_i in os.listdir(path):
        yield  file_i,f'{path}/{file_i}'

def filtered_files(path,taboo):
    if(type(taboo)==str):
        taboo=[taboo]
    if(type(taboo)==list):
        taboo=set(taboo)
    paths=[]
    for file_i in os.listdir(path):
        if(not file_i in taboo):
            paths.append(f'{path}/{file_i}')
    return paths