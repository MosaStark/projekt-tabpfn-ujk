from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import svm
from tabpfn import TabPFNClassifier # Nowy import dla TabPFN

class Clf(object):
    def fit(self,X,y):
        return self.model.fit(X,y)

    def predict(self,X):
        return self.model.predict(X)
    
    def __str__(self):
        return self.NAME
    
    def __repr__(self):
        return self.NAME

class RF(Clf):
    NAME="RF"
    def __init__(self):
        self.model=RandomForestClassifier(class_weight="balanced") 

class GRAD(Clf):
    NAME="GRAD"
    def __init__(self):
        self.model=GradientBoostingClassifier()

class LR(Clf):
    NAME="LR"
    def __init__(self):
        # Poprawiony solver dla klasyfikacji wieloklasowej
        self.model=LogisticRegression(solver='lbfgs', max_iter=1000)

class SVM(Clf):
    NAME="SVM"
    def __init__(self):
        self.model=svm.SVC(kernel='rbf')

class TABPFN(Clf):
    NAME="TabPFN"
    def __init__(self):
        self.model = TabPFNClassifier(device='cuda')