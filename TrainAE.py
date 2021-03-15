import numpy as np
import torch
class AEDataset(torch.utils.data.Dataset):#需要继承data.Dataset
    def __init__(self,Datapath,transform):
        # 1. Initialize file path or list of file names.
        self.Datalist=np.load(Datapath)
        self.transform=transform
    def __getitem__(self, index):
        # 1. Read one data from file (e.g. using numpy.fromfile, PIL.Image.open).
        # 2. Preprocess the data (e.g. torchvision.Transform).
        # 3. Return a data pair (e.g. image and label).
        #这里需要注意的是，第一步：read one data，是一个data
        Data=self.transform(self.Datalist[index].astype('float64'))
        Data=Data.view(1,Data.shape[0],Data.shape[1],Data.shape[2])
        return Data
    def __len__(self):
        # You should change 0 to the total size of your dataset.
        return len(self.Datalist)
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
def generate_(batch_size,dim):
    return torch.from_numpy(
        np.random.multivariate_normal(mean=np.zeros([dim ]), cov=np.diag(np.ones([dim])),
                                      size=batch_size)).type(torch.float)
def TrainAAE_patch(XPath,Patch_channel=15,windowSize=25,encoded_dim=64,batch_size=128,output_units=16):
    import torch
    from DefinedModels import Dec_AAE, Enc_AAE, Discriminant
    from torchvision import transforms
    import numpy as np
    from tqdm import tqdm
    trans = transforms.Compose(transforms=[
        transforms.ToTensor(),
        transforms.Normalize(np.zeros(Patch_channel), np.ones(Patch_channel))
    ])
    Enc_patch = Enc_AAE(channel=Patch_channel,output_dim=encoded_dim,windowSize=windowSize,output_units=output_units).cuda()
    Dec_patch = Dec_AAE(channel=Patch_channel,windowSize=windowSize,input_dim=encoded_dim).cuda()
    discriminant = Discriminant(encoded_dim).cuda()

    ##训练patchAE
    patch_data = AEDataset(XPath,trans)
    Patch_loader = torch.utils.data.DataLoader(dataset=patch_data, batch_size=batch_size, shuffle=True)
    optim_enc = torch.optim.Adam(Enc_patch.parameters(), lr=1e-3, weight_decay=0.0005)
    optim_dec=torch.optim.Adam(Dec_patch.parameters(), lr=1e-3, weight_decay=0.0005)
    optim_enc_gen = torch.optim.SGD(Enc_patch.parameters(), lr=1e-4, weight_decay=0.000)  # 1e-5
    optim_disc = torch.optim.SGD(discriminant.parameters(), lr=5e-5, weight_decay=0.000)  # 5e-6
    criterion = torch.nn.MSELoss()
    epochs=20
    for epoch in range(epochs):
        rl=0
        l_dis_loss=0
        l_encl=0
        print('Epoch No {}'.format(epoch))
        for i, (data) in enumerate(tqdm(Patch_loader)):
            #########################reconstruction phase
            data=data.cuda().float()
            Enc_patch.train()
            Dec_patch.train()
            optim_dec.zero_grad()
            optim_enc.zero_grad()
            optim_disc.zero_grad()
            optim_enc_gen.zero_grad()
            code, out =Enc_patch(data)
            recon=Dec_patch(code)
            loss=criterion(data,recon)
            loss.backward(retain_graph=True)
            optim_dec.step()
            optim_enc.step()
##################################################regularization phase
            discriminant.train()
            Enc_patch.eval()
            gauss=torch.FloatTensor(generate_(batch_size,encoded_dim)).cuda()
            fake_pred = discriminant(gauss)
            true_pred = discriminant(code)
            dis_loss=-(torch.mean(fake_pred) -torch.mean(true_pred))
            dis_loss.backward(retain_graph=True)
            optim_disc.step()
            discriminant.train()
            Enc_patch.train()
            encl=-torch.mean(true_pred)
            encl.backward(retain_graph=True)
            optim_enc_gen.step()
            rl = rl + loss.item()
            l_dis_loss+=dis_loss.item()
            l_encl+=encl.item()
        print('\nPatch Reconstruction Loss: {}  dis loss: {}   regularization loss : {}'.format(rl/len(patch_data),l_dis_loss/len(patch_data),l_encl/len(patch_data)))
    torch.save(Enc_patch.state_dict(),'./models/Pretrained.pth')
    return 0
def SaveFeatures_AAE(modelpath,XPath,Patch_channel=15,windowSize=25,encoded_dim=64,batch_size=128, output_units=16):
    import torch
    from DefinedModels import  Enc_AAE
    from torchvision import transforms
    import numpy as np
    from tqdm import tqdm
    from Preprocess import feature_normalize2, L2_Norm
    trans = transforms.Compose(transforms=[
        transforms.ToTensor(),
        transforms.Normalize(np.zeros(Patch_channel), np.ones(Patch_channel))
    ])
    Enc_patch = Enc_AAE(channel=Patch_channel,output_dim=encoded_dim,windowSize=windowSize, output_units=output_units).cuda()
    Enc_patch.load_state_dict(torch.load(modelpath))
    ##运行patchAE 的encoder
    patch_data = AEDataset(XPath,trans)
    Patch_loader = torch.utils.data.DataLoader(dataset=patch_data, batch_size=batch_size, shuffle=False)
    Patch_Features=[]
    print('Start save patch features...')
    for i, (data) in enumerate(tqdm(Patch_loader)):
        data=data.cuda().float()
        Enc_patch.eval()
        feature, out= Enc_patch(data)
        for num in range(len(feature)):
            Patch_Features.append(np.array(feature[num].cpu().detach().numpy()))
    # Patch_Features=feature_normalize2(Patch_Features)
    # Patch_Features = L2_Norm(Patch_Features)

    return Patch_Features
