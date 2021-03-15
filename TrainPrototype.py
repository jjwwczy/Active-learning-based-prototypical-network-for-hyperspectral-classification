import torch
import numpy as np
# -*- coding: utf-8 -*-
def distEclud(vecA, vecB):
    '''
    输入：向量A和B
    输出：A和B间的欧式距离
    '''
    return np.sqrt(sum(np.power(vecA - vecB, 2)))
def newCent(RequestX, RequestY):
    '''
    输入：有标签数据集L
    输出：根据L确定初始聚类中心
    '''
    centroids = []
    label_list = np.unique(RequestY)
    for i in label_list:
        L_i = RequestX[RequestY== i]
        cent_i = np.mean(L_i, 0)
        centroids.append(cent_i)
    return np.array(centroids)
def FindMin(temp_distance,temp_index):
    min_id = np.argmin(temp_distance)
    global_id = temp_index[min_id]
    return min_id,global_id
def Search(centroids, dataSet, clusterAssment):
    #中心，数据，和数据对应的中心
    D=np.zeros(len(clusterAssment))
    for i in range(len(D)):
        D[i]=distEclud(centroids[int(clusterAssment[i])], dataSet[i])
    return D, clusterAssment
def semi_kMeans(features,CurrentArray, RequestX, RequestY):
    '''
    输入：有标签数据集L（最后一列为类别标签）、无标签数据集U（无类别标签）
    输出：聚类结果
    '''
    dataSet = features
    label_list = np.unique(RequestY)
    k = len(label_list)  # L中类别个数
    m = np.shape(dataSet)[0]

    clusterAssment = np.zeros(m)  # 初始化样本的分配
    centroids = newCent(RequestX, RequestY)  # 确定初始聚类中心
    clusterChanged = True
    while clusterChanged:
        clusterChanged = False
        for i in range(m):  # 将每个样本分配给最近的聚类中心
            minDist = np.inf;
            minIndex = -1
            for j in range(k):
                distJI = distEclud(centroids[j], dataSet[i])
                if distJI < minDist:
                    minDist = distJI;
                    minIndex = j
            if clusterAssment[i] != minIndex: clusterChanged = True
            clusterAssment[i] = int(minIndex)
        for cent in range(k):
            # 得到属于第cent个簇的样本的集合
            ptsInClust = dataSet[np.nonzero(clusterAssment == cent)]
            # 计算这个集合里面样本的均值，即中心
            centroids[cent] = np.mean(ptsInClust, axis=0)
    D, I = Search(centroids, dataSet, clusterAssment)  # for each sample, find cluster distance and assignments
    Id=np.arange(0,len(dataSet))
    RequestArray = []
    for i in range(k):
        temp_distance=D[I==i]
        temp_index=Id.reshape(-1, 1)[I==i]
        temp_index = temp_index.reshape(-1)

        while True:
            min_id ,global_id = FindMin(temp_distance, temp_index)
            if global_id not in CurrentArray:
                break
            else:
                temp_distance=np.delete(temp_distance,min_id)
                temp_index = np.delete(temp_index, min_id)
        RequestArray.append(global_id)
    return RequestArray


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


def SelectSamples(AAEPath,CurrentArray,RequestY):
    features=np.load(AAEPath)
    RequestArray=semi_kMeans(features,CurrentArray, features[CurrentArray], RequestY)
    return RequestArray


def TrainWholeNet(Enc_patch,RequestXPath,RequestYPath, Patch_channel):
    import torch
    from torchvision import transforms
    import numpy as np
    import os
    import itertools
    trans = transforms.Compose(transforms=[
        transforms.ToTensor(),
        transforms.Normalize(np.zeros(Patch_channel), np.ones(Patch_channel))
    ])
    s_data = MYDataset(RequestXPath,RequestYPath,trans)
    s_loader=torch.utils.data.DataLoader(dataset=s_data, batch_size=256, shuffle=True)
    optim_classi = torch.optim.Adam(Enc_patch.parameters(), lr=1e-3, weight_decay=1e-06)
    criterion = torch.nn.CrossEntropyLoss()
    epoch=50
    print("数据集长度："+str(len(s_data)))
    for i in range(epoch):
        epoch_classiloss = 0.
        train_acc = 0.
        for batch_idx, (x,y) in enumerate(s_loader):
            x = x.cuda()
            y = y.cuda()
            Enc_patch.train()
            feature, output = Enc_patch(x)
            classi_loss = criterion(output, y)
            epoch_classiloss += classi_loss.item()
            pred = torch.max(output, 1)[1]
            train_correct = (pred == y).sum()
            train_acc += train_correct.item()
            optim_classi.zero_grad()
            classi_loss.backward(retain_graph=False)
            optim_classi.step()
        print(
            'Train Loss: {:.6f}, Acc: {:.6f}'.format(classi_loss / (len(s_data)), train_acc / (len(s_data))))

    torch.save(Enc_patch.state_dict(), './models/Trained.pth')
    return 0

def TestWholeNet(Enc_patch,XPath,YPath, Patch_channel):
    import torch
    from torchvision import transforms
    import numpy as np
    import os
    trans = transforms.Compose(transforms=[
        transforms.ToTensor(),
        transforms.Normalize(np.zeros(Patch_channel), np.ones(Patch_channel))
    ])
    s_data = MYDataset(XPath, YPath, trans)
    s_loader = torch.utils.data.DataLoader(s_data, 256, shuffle=False)
    Result=[]
    for batch_idx, (x, y) in enumerate(s_loader):
        Enc_patch.eval()
        x = x.cuda()
        y = y.cuda()
        feature, output = Enc_patch(x)
        for num in range(len(output)):
            Result.append(np.array(output[num].cpu().detach().numpy()))
    return Result