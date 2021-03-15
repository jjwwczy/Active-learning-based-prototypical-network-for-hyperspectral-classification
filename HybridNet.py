import torch
import torch.nn as nn
from torchvision import transforms
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, cohen_kappa_score
from operator import truediv
import numpy as np
import scipy.io as sio
import os
import spectral
from torchsummary import summary
import copy
import time
import argparse
import csv
class ConvBNRelu3D(nn.Module):
    def __init__(self,in_channels=1, out_channels=24, kernel_size=(51, 3, 3), padding=0,stride=1):
        super(ConvBNRelu3D,self).__init__()
        self.in_channels=in_channels
        self.out_channels=out_channels
        self.kernel_size=kernel_size
        self.padding=padding
        self.stride=stride
        self.conv=nn.Conv3d(in_channels=self.in_channels, out_channels=self.out_channels, kernel_size=self.kernel_size, stride=self.stride,padding=self.padding)
        self.bn=nn.BatchNorm3d(num_features=self.out_channels)
        self.relu = nn.ReLU(inplace=False)
    def forward(self,x):
        x = self.conv(x)
        x = self.bn(x)
        x= self.relu(x)
        return x
class ConvBNRelu2D(nn.Module):
    def __init__(self,in_channels=1, out_channels=24, kernel_size=(51, 3, 3), stride=1,padding=0):
        super(ConvBNRelu2D,self).__init__()
        self.stride = stride
        self.in_channels=in_channels
        self.out_channels=out_channels
        self.kernel_size=kernel_size
        self.padding=padding
        self.conv=nn.Conv2d(in_channels=self.in_channels, out_channels=self.out_channels, kernel_size=self.kernel_size, stride=self.stride,padding=self.padding)
        self.bn=nn.BatchNorm2d(num_features=self.out_channels)
        self.relu = nn.ReLU(inplace=False)
    def forward(self,x):
        x = self.conv(x)
        x = self.bn(x)
        x= self.relu(x)
        return x
class HybridNet(nn.Module):
    def __init__(self,channel,output_units,windowSize):
        # 调用Module的初始化
        super(HybridNet, self).__init__()
        self.channel=channel
        self.output_units=output_units
        self.windowSize=windowSize
        self.conv1 = ConvBNRelu3D(in_channels=1,out_channels=8,kernel_size=(7,3,3),stride=1,padding=0)
        self.conv2 = ConvBNRelu3D(in_channels=8,out_channels=16,kernel_size=(5,3,3),stride=1,padding=0)
        self.conv3 = ConvBNRelu3D(in_channels=16,out_channels=32,kernel_size=(3,3,3),stride=1,padding=0)
        self.conv4 = ConvBNRelu2D(in_channels=32*(self.channel-12), out_channels=64, kernel_size=(3, 3), stride=1, padding=0)
        self.fc1=nn.Linear(in_features=64*(self.windowSize-8)*(self.windowSize-8),out_features=256)
        self.relu1 = nn.ReLU(inplace=False)
        # self.drop1=nn.Dropout2d(0.4)
        self.fc2 = nn.Linear(in_features=256, out_features=128)
        self.relu2 = nn.ReLU(inplace=False)
        # self.drop2 = nn.Dropout2d(0.4)
        self.fc3 = nn.Linear(in_features=128, out_features=self.output_units)
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.reshape([x.shape[0],-1,x.shape[3],x.shape[4]])
        x = self.conv4(x)
        x = x.reshape([x.shape[0],-1])
        x = self.fc1(x)
        x=self.relu1(x)
        # x=self.drop1(x)
        x = self.fc2(x)
        x=self.relu2(x)

        # x= self.drop2(x)
        x = self.fc3(x)
        return x
def loadData(name):
    data_path = os.path.join(os.getcwd(), 'datasets')
    if name == 'IP':
        data = sio.loadmat(os.path.join(data_path, 'Indian_pines_corrected.mat'))['indian_pines_corrected']
        labels = sio.loadmat(os.path.join(data_path, 'Indian_pines_gt.mat'))['indian_pines_gt']
    elif name == 'SA':
        data = sio.loadmat(os.path.join(data_path, 'Salinas_corrected.mat'))['salinas_corrected']
        labels = sio.loadmat(os.path.join(data_path, 'Salinas_gt.mat'))['salinas_gt']
    elif name == 'PU':
        data = sio.loadmat(os.path.join(data_path, 'PaviaU.mat'))['paviaU']
        labels = sio.loadmat(os.path.join(data_path, 'PaviaU_gt.mat'))['paviaU_gt']
    elif name == 'KSC':
        data = sio.loadmat(os.path.join(data_path, 'KSC.mat'))['KSC']
        labels = sio.loadmat(os.path.join(data_path, 'KSC_gt.mat'))['KSC_gt']
    return data, labels

def PerClassSplit(X, y, perclass, stratify):
    X_train=[]
    y_train=[]
    X_test = []
    y_test = []
    for label in stratify:
        indexList = [i for i in range(len(y)) if y[i] == label]
        # print(len(indexList))
        train_index=np.random.choice(indexList,perclass,replace=True)
        for i in range(len(train_index)):
            index=train_index[i]
            X_train.append(X[index])
            y_train.append(label)
        test_index = [i for i in indexList if i not in train_index]
        # temp=len(y_test)
        for i in range(len(test_index)):
            index=test_index[i]
            X_test.append(X[index])
            y_test.append(label)
        # print(len(y_test)-temp)
    return X_train, X_test, y_train, y_test
def splitTrainTestSet(X, y, testRatio, randomState=345):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=testRatio, random_state=randomState,
                                                        stratify=y)
    return X_train, X_test, y_train, y_test


def applyPCA(X, numComponents=75):
    newX = np.reshape(X, (-1, X.shape[2]))
    pca = PCA(n_components=numComponents, whiten=True)
    newX = pca.fit_transform(newX)
    newX = np.reshape(newX, (X.shape[0], X.shape[1], numComponents))
    return newX, pca


def padWithZeros(X, margin=2):
    newX = np.zeros((X.shape[0] + 2 * margin, X.shape[1] + 2 * margin, X.shape[2]))
    x_offset = margin
    y_offset = margin
    newX[x_offset:X.shape[0] + x_offset, y_offset:X.shape[1] + y_offset, :] = X
    return newX


def createImageCubes(X, y, windowSize=5, removeZeroLabels=True):
    margin = int((windowSize - 1) / 2)
    zeroPaddedX = padWithZeros(X, margin=margin)
    # split patches
    patchesData = np.zeros((X.shape[0] * X.shape[1], windowSize, windowSize, X.shape[2]), dtype=np.float32)
    patchesLabels = np.zeros((X.shape[0] * X.shape[1]), dtype=np.float32)
    patchIndex = 0
    for r in range(margin, zeroPaddedX.shape[0] - margin):
        for c in range(margin, zeroPaddedX.shape[1] - margin):
            patch = zeroPaddedX[r - margin:r + margin + 1, c - margin:c + margin + 1]
            patchesData[patchIndex, :, :, :] = patch
            patchesLabels[patchIndex] = y[r - margin, c - margin]
            patchIndex = patchIndex + 1
    if removeZeroLabels:
        patchesData = patchesData[patchesLabels > 0, :, :, :]
        patchesLabels = patchesLabels[patchesLabels > 0]
        patchesLabels -= 1
    return patchesData, patchesLabels

class MYDataset(torch.utils.data.Dataset):#需要继承data.Dataset
    def __init__(self,Datapath,Labelpath,transform):
        # 1. Initialize file path or list of file names.
        self.Datalist=np.load(Datapath)
        self.Labellist=(np.load(Labelpath)).astype(int)
        self.transform=transform
    def __getitem__(self, index):
        # 1. Read one data from file (e.g. using numpy.fromfile, PIL.Image.open).
        # 2. Preprocess the data (e.g. torchvision.Transform).
        # 3. Return a data pair (e.g. image and label).
        #这里需要注意的是，第一步：read one data，是一个data

        index=index
        Data=self.transform(self.Datalist[index])
        Data=Data.view(1,Data.shape[0],Data.shape[1],Data.shape[2])
        return Data ,self.Labellist[index]
    def __len__(self):
        # You should change 0 to the total size of your dataset.
        return len(self.Datalist)
def train(model,Datapath,Labelpath):

    train_data = MYDataset(Datapath,Labelpath,trans)
    train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=256, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(),lr=0.001,weight_decay=1e-06)
    criterion=torch.nn.CrossEntropyLoss()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    Best_loss=10000.0
    for epoch in range(70):
        print('epoch {}'.format(epoch + 1))
        # training-----------------------------
        train_loss = 0.
        train_acc = 0.
        model.train()
        model = model.cuda()
        for data, label in train_loader:
            data=data.cuda()
            out = model(data)
            label=label.cuda()
            loss = criterion(out, label)
            train_loss += loss.item()
            pred = torch.max(out, 1)[1]
            train_correct = (pred ==label).sum()
            train_acc += train_correct.item()
            optimizer.zero_grad()
            loss.backward(retain_graph=False)
            optimizer.step()
        print('Train Loss: {:.6f}, Acc: {:.6f}'.format(train_loss / (len(train_data)), train_acc / (len(train_data))))
        if (train_acc / (len(train_data))>=best_acc) and (train_loss / (len(train_data))<Best_loss):
            best_model_wts = copy.deepcopy(model.state_dict())

    # torch.save(best_model_wts,'model.pth')
    torch.save(model.state_dict(), 'model.pth')
    return 0

def predict(model,Datapath,Labelpath):
    model.eval()
    model = model.cuda()
    test_data = MYDataset(Datapath,Labelpath,trans)
    test_loader = torch.utils.data.DataLoader(dataset=test_data, batch_size=128, shuffle=False)
    prediction=[]
    for data, label in test_loader:
        data=data.cuda()
        out = model(data)
        for num in range(len(out)):
            prediction.append(np.array(out[num].cpu().detach().numpy()))
    return prediction
def evaluate(model,Datapath,Labelpath):
    model.eval()
    model = model.cuda()
    criterion = torch.nn.CrossEntropyLoss()
    test_data = MYDataset(Datapath,Labelpath,trans)
    test_loader = torch.utils.data.DataLoader(dataset=test_data, batch_size=128, shuffle=False)
    score=np.zeros(2)
    train_loss=0.
    train_acc=0.
    index=0
    for data, label in test_loader:
        data=data.cuda()
        out = model(data)
        label = label.cuda()
        loss = criterion(out, label)
        train_loss += loss.item()
        pred = torch.max(out, 1)[1]
        train_correct = (pred ==label).sum()
        train_acc += train_correct.item()
    score[0]=train_loss/ (len(test_data))
    score[1]=train_acc/ (len(test_data))
    return score
def AA_andEachClassAccuracy(confusion_matrix):
    counter = confusion_matrix.shape[0]
    list_diag = np.diag(confusion_matrix)
    list_raw_sum = np.sum(confusion_matrix, axis=1)
    each_acc = np.nan_to_num(truediv(list_diag, list_raw_sum))
    average_acc = np.mean(each_acc)
    return each_acc, average_acc


def reports(model,X_test, y_test, name):
    # start = time.time()
    Datapath='Xtest.npy'
    Labelpath='ytest.npy'
    Y_pred = predict(model,Datapath,Labelpath)
    y_pred = np.argmax(np.array(Y_pred), axis=1)
    # end = time.time()
    # print(end - start)
    if name == 'IP':
        target_names = ['Alfalfa', 'Corn-notill', 'Corn-mintill', 'Corn'
            , 'Grass-pasture', 'Grass-trees', 'Grass-pasture-mowed',
                        'Hay-windrowed', 'Oats', 'Soybean-notill', 'Soybean-mintill',
                        'Soybean-clean', 'Wheat', 'Woods', 'Buildings-Grass-Trees-Drives',
                        'Stone-Steel-Towers']
    elif name == 'SA':
        target_names = ['Brocoli_green_weeds_1', 'Brocoli_green_weeds_2', 'Fallow', 'Fallow_rough_plow',
                        'Fallow_smooth',
                        'Stubble', 'Celery', 'Grapes_untrained', 'Soil_vinyard_develop', 'Corn_senesced_green_weeds',
                        'Lettuce_romaine_4wk', 'Lettuce_romaine_5wk', 'Lettuce_romaine_6wk', 'Lettuce_romaine_7wk',
                        'Vinyard_untrained', 'Vinyard_vertical_trellis']
    elif name == 'PU':
        target_names = ['Asphalt', 'Meadows', 'Gravel', 'Trees', 'Painted metal sheets', 'Bare Soil', 'Bitumen',
                        'Self-Blocking Bricks', 'Shadows']
    elif name == 'KSC':
        target_names = ["Scrub", "Willow swamp",
                        "Cabbage palm hammock", "Cabbage palm/oak hammock",
                        "Slash pine", "Oak/broadleaf hammock",
                        "Hardwood swamp", "Graminoid marsh", "Spartina marsh",
                        "Cattail marsh", "Salt marsh", "Mud flats", "Wate"]
    classification = classification_report(y_test, y_pred, target_names=target_names)
    oa = accuracy_score(y_test, y_pred)
    confusion = confusion_matrix(y_test, y_pred)
    each_acc, aa = AA_andEachClassAccuracy(confusion)
    kappa = cohen_kappa_score(y_test, y_pred)
    score = evaluate(model,Datapath,Labelpath)
    Test_Loss = score[0] * 100
    Test_accuracy = score[1] * 100
    return classification, confusion, Test_Loss, Test_accuracy, oa * 100, each_acc * 100, aa * 100, kappa * 100


dataset_names = ['IP','SA','PU']
parser = argparse.ArgumentParser(description="Run deep learning experiments on"
                                             " various hyperspectral datasets")
parser.add_argument('--dataset', type=str, default='SA', choices=dataset_names,
                    help="Dataset to use.")
parser.add_argument('--train',type=bool, default=0)
parser.add_argument('--perclass', type=int, default=3)
args = parser.parse_args()
TRAIN =args.train

dataset = args.dataset
perclass=args.perclass

output_units = 9 if (dataset == 'PU' or dataset == 'PC') else 16
#IP 10249   PU 42776   SA  54129
#
windowSize = 25
X, y = loadData(dataset)
K = X.shape[2]
K = 30 if dataset == 'IP' else 15
trans = transforms.Compose(transforms = [
    transforms.ToTensor(),
    transforms.Normalize(np.zeros(K),np.ones(K))
])
X, pca = applyPCA(X, numComponents=K)
X, y = createImageCubes(X, y, windowSize=windowSize)
def feature_normalize(data):
    mu = np.mean(data,axis=0)
    std = np.std(data,axis=0)
    return (data - mu)/std
X=feature_normalize(X)

stratify=np.arange(0,output_units,1)
if perclass > 1:
    Xtrain, Xtest, ytrain, ytest = PerClassSplit(X, y, args.perclass, stratify)
else:
    Xtrain, Xtest, ytrain, ytest = splitTrainTestSet(X, y, 1 - args.perclass,randomState=345)
# Xtrain = Xtrain.reshape(-1,1,K,windowSize, windowSize)
# Xtest = Xtest.reshape(-1, 1, K, windowSize, windowSize)
np.save('Xtrain.npy',Xtrain)
np.save('ytrain.npy',ytrain)
np.save('Xtest.npy',Xtest)
np.save('ytest.npy',ytest)
model=HybridNet(channel=K,output_units=output_units,windowSize=windowSize)
model = model.cuda()
# summary(model,(1,K,windowSize,windowSize))
Datapath='Xtrain.npy'
Labelpath='ytrain.npy'
train_start_time=time.time()
if TRAIN:
    train(model,Datapath,Labelpath)
train_end_time=time.time()
model.load_state_dict(torch.load('model.pth'))
Datapath='Xtest.npy'
Labelpath='ytest.npy'
test_start_time=time.time()
Y_pred_test = predict(model,Datapath,Labelpath)
y_pred_test = np.argmax(np.array(Y_pred_test), axis=1)

for i in y_pred_test:
    csvFile = open("Good.csv", "a")
    writer = csv.writer(csvFile)
    writer.writerow(str(i))
csvFile.close()
test_end_time=time.time()
classification = classification_report(ytest, y_pred_test)
print(classification)
print()
print('training time:',train_end_time-train_start_time,'s')
print()
print('testing time:',test_end_time-test_start_time,'s')
