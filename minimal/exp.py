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