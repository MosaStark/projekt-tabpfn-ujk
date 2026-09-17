from dataclasses import dataclass#,field
import base
import make_results
import dataset

@dataclass
class ExpParams:
    data_path:str      
    split_path:str
    out_path:str = None
    clf_type:str = 'RF' 
    k:int = 100 
    
    @property
    def id(self):
        return self.data_path.split("/")[-1]

    def get_data(self):
        data=dataset.read_csv(self.data_path)
        splits=base.SplitGroup.read(self.split_path)
        return data,splits

    def get_clf(self):
        return make_results.CLF_DICT[self.clf_type]

    def iter_exp(self,attr,values):
        for value_i in values:
            exp_i = ExpParams(**self.__dict__)
            setattr(exp_i, attr, value_i)
            yield exp_i

def from_dir( data_path,
              split_path,
              clfs):
    exps=[]
    for id_i,data_i in base.iter_files(data_path):
        split_i=f"{split_path}/{id_i}"
        exp_i=ExpParams( data_i,
                         split_i,
                         clfs)
        exps.append(exp_i)
    return exps
