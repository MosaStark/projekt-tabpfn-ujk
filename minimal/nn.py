from tabpfn import TabPFNClassifier
import clf

class TabPFN(clf.Clf):
    NAME="TabPFN"
    def __init__( self):
        self.model=TabPFNClassifier()