# 1. 数据预处理，包括读数据，生成数组文件等等
#
import argparse
from Preprocess import Preprocess,PerClassSplit,splitTrainTestSet
from DefinedModels import Enc_AAE
from TrainAE import TrainAAE_patch,SaveFeatures_AAE
from TrainPrototype import SelectSamples,TrainWholeNet,TestWholeNet
import numpy as np
import joblib
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, cohen_kappa_score
import time
import csv
from utils import reports
import torch
import math
dataset_names = ['IP','SA','PU']
parser = argparse.ArgumentParser(description="Run deep learning experiments on"
                                             " various hyperspectral datasets")
parser.add_argument('--dataset', type=str, default='IP', choices=dataset_names,
                    help="Dataset to use.")
parser.add_argument('--train',type=int, default=0,choices=(0,1))
parser.add_argument('--perclass', type=int, default=3) #会除以100
parser.add_argument('--device', type=str, default="cuda:0", choices=("cuda:0","cuda:1"))
parser.add_argument('--projection_dim', type=int, default=128)
parser.add_argument('--encoded_dim', type=int, default=128)
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--Windowsize', type=int, default=15)
parser.add_argument('--Patch_channel', type=int, default=30)
parser.add_argument('--RandomSeed', type=bool, default=False)#True 代表固定随机数， False代表不固定

args = parser.parse_args()

if args.RandomSeed:
    randomState=345
else:
    randomState=int(np.random.randint(1, high=1000))
print(args)
output_units = 9 if (args.dataset == 'PU' or args.dataset == 'PC') else 16
Datadir='./DataArray/'
XPath = Datadir + 'X.npy'
yPath = Datadir + 'y.npy'
# # 2. 生成自编码器并训练，保存好模型参数
train_start = time.time()
Preprocess(XPath, yPath, args.dataset, args.Windowsize, Patch_channel=args.Patch_channel)
if args.train:


    # TrainVAE_patch(XPath, Patch_channel=args.Patch_channel, windowSize=args.Windowsize, encoded_dim=args.encoded_dim,
    #               batch_size=args.batch_size)
    TrainAAE_patch(XPath,Patch_channel=args.Patch_channel,windowSize=args.Windowsize,encoded_dim=args.encoded_dim,batch_size=args.batch_size, output_units=output_units
                   )

#############用聚类算法挑选需要标签的数据，对特征数组做标记，需要标签的置为1，不需要的置0，返回含标记的数组
originalNumber=math.ceil(0.5*args.perclass)
RequestArray=[]
X=np.load(XPath)
Y = np.load(yPath)
stratify=np.arange(0,output_units,1)
Id=np.arange(0,len(X))
for label in stratify:
    temp_index = Id.reshape(-1, 1)[Y == label]
    temp_index=temp_index.reshape(-1)
    train_index = np.random.choice(temp_index, originalNumber, replace=True)
    for i in range(len(train_index)):
        RequestArray.append(train_index[i])
RequestX=X[RequestArray]
RequestXPath=Datadir+'RequestX.npy'
np.save(RequestXPath,RequestX)
RequestY=Y[RequestArray]
RequestYPath=Datadir+'RequestY.npy'
np.save(RequestYPath,RequestY)


model = Enc_AAE(channel=args.Patch_channel,output_dim=args.encoded_dim,windowSize=args.Windowsize,output_units=output_units).cuda()
if not args.train:
    model.load_state_dict(torch.load('./models/Pretrained.pth'))

modelpath='./models/Pretrained.pth'
AAEFeatures=SaveFeatures_AAE(modelpath,XPath,Patch_channel=args.Patch_channel,windowSize=args.Windowsize,encoded_dim=args.encoded_dim,batch_size=args.batch_size, output_units=output_units)
AAEPath=Datadir+'Features.npy'
np.save(AAEPath,AAEFeatures)
for i in range(args.perclass-originalNumber):
    RequestArray=RequestArray+SelectSamples(AAEPath,RequestArray, RequestY)
    X=np.load(XPath)
    RequestX=X[RequestArray]
    RequestXPath=Datadir+'RequestX.npy'
    np.save(RequestXPath,RequestX)
    Y = np.load(yPath)
    RequestY=Y[RequestArray]
    RequestYPath=Datadir+'RequestY.npy'
    np.save(RequestYPath,RequestY)
    TrainWholeNet(model, RequestXPath,RequestYPath, Patch_channel=args.Patch_channel)
    modelpath='./models/Trained.pth'
    AAEFeatures = SaveFeatures_AAE(modelpath,XPath, Patch_channel=args.Patch_channel, windowSize=args.Windowsize,
                                   encoded_dim=args.encoded_dim, batch_size=args.batch_size, output_units=output_units)
    np.save(AAEPath, AAEFeatures)

Total=np.arange(0,len(X))
Xindex=[x for x in Total if x not in RequestArray]
yindex=[y for y in Total if y not in RequestArray]
Xtest=X[Xindex]
ytest=Y[yindex]
np.save(Datadir + 'Xtest.npy', Xtest)
np.save(Datadir + 'ytest.npy', ytest)
model.load_state_dict(torch.load('./models/Trained.pth'))
Prediction=TestWholeNet(model,XPath=Datadir + 'Xtest.npy',YPath=Datadir + 'ytest.npy', Patch_channel=args.Patch_channel)
Predictions = np.argmax(np.array(Prediction), axis=1)
ytest=np.load(Datadir + 'ytest.npy')
classification = classification_report(ytest.astype(int), Predictions)
print(classification)
classification, confusion, oa, each_acc, aa, kappa = reports(Predictions, ytest.astype(int), args.dataset)
## 9. 存储分类报告至csv，以及画出效果图
# file_name = args.dataset+'_'+str(args.perclass)+"perclass.txt"
# with open(file_name, 'w') as x_file:
#     x_file.write('{}'.format(classification))
each_acc_str = ','.join(str(x) for x in each_acc)
add_info=[args.dataset,args.perclass,args.Windowsize,oa,aa,kappa]+each_acc_str.split('[')[0].split(']')[0].split(',')
csvFile = open("ActiveLearning2.csv", "a")
writer = csv.writer(csvFile)
writer.writerow(add_info)
csvFile.close()
torch.cuda.empty_cache()