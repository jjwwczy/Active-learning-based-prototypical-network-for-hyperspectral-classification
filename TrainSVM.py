import sklearn
from sklearn.neighbors import KNeighborsClassifier
import torch
import copy
import numpy as np
from tqdm import tqdm
def TrainSVM(Xtrain,ytrain):
    SVM_GRID_PARAMS = [
        {'kernel': ['rbf'], 'gamma': [0.001, 0.01, 0.1, 1], 'C': [0.1, 1, 10, 100, 1000]},
        {'kernel': ['linear'], 'C': [0.1, 1, 10, 100, 1000]},
        {'kernel': ['poly'], 'degree': [3], 'gamma': [1e-1, 1e-2, 1e-3]}
    ]
    class_weight = 'balanced'
    clf = sklearn.svm.SVC(class_weight=class_weight,probability=True,gamma='scale',kernel='linear')
    clf = sklearn.model_selection.GridSearchCV(clf, SVM_GRID_PARAMS, scoring=None, n_jobs=-1, iid=True,
                                              refit=True, cv=3, verbose=3, pre_dispatch='2*n_jobs',
                                              error_score='raise', return_train_score=True)
    clf.fit(Xtrain, ytrain)
    print(clf.best_params_)
    return clf
def TestSVM(Xtest,clf):

    Prediction=clf.predict(Xtest)
    return Prediction
class MYDataset(torch.utils.data.Dataset):#需要继承data.Dataset
    def __init__(self,Datapath,Labelpath):
        # 1. Initialize file path or list of file names.
        self.Datalist=np.load(Datapath)
        self.Labellist=(np.load(Labelpath)).astype(int)
    def __getitem__(self, index):
        # 1. Read one data from file (e.g. using numpy.fromfile, PIL.Image.open).
        # 2. Preprocess the data (e.g. torchvision.Transform).
        # 3. Return a data pair (e.g. image and label).
        #这里需要注意的是，第一步：read one data，是一个data

        index=index
        Data=torch.FloatTensor(self.Datalist[index])
        return Data ,self.Labellist[index]
    def __len__(self):
        # You should change 0 to the total size of your dataset.
        return len(self.Datalist)
def TrainKNN(template_features,output_units):
    neigh = KNeighborsClassifier(n_neighbors=1)
    X=template_features
    y=np.arange(0,output_units,1)
    neigh.fit(X, y)

    return neigh
def TestKNN(model,vae_features):
    Features=model.predict(vae_features)
    return Features