import matplotlib.pyplot as plt
import torch

from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, cohen_kappa_score
from operator import truediv
import numpy as np
import scipy.io as sio
import os
import time
import csv
import argparse
apex = False
import seaborn as sns
import pandas as pd
import random
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
def PerClassSplit(X, y, perclass, stratify,randomState=345):
    np.random.seed(randomState)
    X_train=[]
    y_train=[]
    X_test = []
    y_test = []
    for label in stratify:

        indexList = [i for i in range(len(y)) if y[i] == label]
        train_index=np.random.choice(indexList,perclass,replace=True)
        for i in range(len(train_index)):
            index=train_index[i]
            X_train.append(X[index])
            y_train.append(label)
        test_index = [i for i in indexList if i not in train_index]

        for i in range(len(test_index)):
            index=test_index[i]
            X_test.append(X[index])
            y_test.append(label)
    return X_train, X_test, y_train, y_test
def AA_andEachClassAccuracy(confusion_matrix):
    counter = confusion_matrix.shape[0]
    list_diag = np.diag(confusion_matrix)
    list_raw_sum = np.sum(confusion_matrix, axis=1)
    each_acc = np.nan_to_num(truediv(list_diag, list_raw_sum))
    average_acc = np.mean(each_acc)
    return each_acc, average_acc
def reports(y_pred,y_test, name):
    # start = time.time()

    # end = time.time()
    # print(end - start)
    Label=y_test
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

    classification = classification_report(Label, y_pred, target_names=target_names)
    oa = accuracy_score(Label, y_pred)
    confusion = confusion_matrix(Label, y_pred)
    each_acc, aa = AA_andEachClassAccuracy(confusion)
    kappa = cohen_kappa_score(Label, y_pred)
    return classification, confusion, oa * 100, each_acc * 100, aa * 100, kappa, target_names

from skimage.morphology import reconstruction
from skimage.morphology import erosion
from skimage.morphology import disk
from skimage import util
def opening_by_reconstruction(image, se):
    """
        Performs an Opening by Reconstruction.

        Parameters:
            image: 2D matrix.
            se: structuring element
        Returns:
            2D matrix of the reconstructed image.
    """
    eroded = erosion(image, se)
    reconstructed = reconstruction(eroded, image)
    return reconstructed
def closing_by_reconstruction(image, se):
    """
        Performs a Closing by Reconstruction.

        Parameters:
            image: 2D matrix.
            se: structuring element
        Returns:
            2D matrix of the reconstructed image.
    """
    obr = opening_by_reconstruction(image, se)

    obr_inverted = util.invert(obr)
    obr_inverted_eroded = erosion(obr_inverted, se)
    obr_inverted_eroded_rec = reconstruction(
        obr_inverted_eroded, obr_inverted)
    obr_inverted_eroded_rec_inverted = util.invert(obr_inverted_eroded_rec)
    return obr_inverted_eroded_rec_inverted
def build_morphological_profiles(image, se_size=4, se_size_increment=2, num_openings_closings=4):
    """
        Build the morphological profiles for a given image.

        Parameters:
            base_image: 2d matrix, it is the spectral information part of the MP.
            se_size: int, initial size of the structuring element (or kernel). Structuring Element used: disk
            se_size_increment: int, structuring element increment step
            num_openings_closings: int, number of openings and closings by reconstruction to perform.
        Returns:
            emp: 3d matrix with both spectral (from the base_image) and spatial information
    """
    x, y = image.shape
    cbr = np.zeros(shape=(x, y, num_openings_closings))
    obr = np.zeros(shape=(x, y, num_openings_closings))

    it = 0
    tam = se_size
    while it < num_openings_closings:
        se = disk(tam)
        temp = closing_by_reconstruction(image, se)
        cbr[:, :, it] = temp[:, :]
        temp = opening_by_reconstruction(image, se)
        obr[:, :, it] = temp[:, :]
        tam += se_size_increment
        it += 1
    mp = np.zeros(shape=(x, y, (num_openings_closings*2)+1))
    cont = num_openings_closings - 1
    for i in range(num_openings_closings):
        mp[:, :, i] = cbr[:, :, cont]
        cont = cont - 1
    mp[:, :, num_openings_closings] = image[:, :]
    cont = 0
    for i in range(num_openings_closings+1, num_openings_closings*2+1):
        mp[:, :, i] = obr[:, :, cont]
        cont += 1

    return mp
def build_emp(base_image, se_size=4, se_size_increment=2, num_openings_closings=4):
    """
        Build the extended morphological profiles for a given set of images.

        Parameters:
            base_image: 3d matrix, each 'channel' is considered for applying the morphological profile. It is the spectral information part of the EMP.
            se_size: int, initial size of the structuring element (or kernel). Structuring Element used: disk
            se_size_increment: int, structuring element increment step
            num_openings_closings: int, number of openings and closings by reconstruction to perform.
        Returns:
            emp: 3d matrix with both spectral (from the base_image) and spatial information
    """
    base_image_rows, base_image_columns, base_image_channels = base_image.shape
    se_size = se_size
    se_size_increment = se_size_increment
    num_openings_closings = num_openings_closings
    morphological_profile_size = (num_openings_closings * 2) + 1
    emp_size = morphological_profile_size * base_image_channels
    emp = np.zeros(
        shape=(base_image_rows, base_image_columns, emp_size))

    cont = 0
    for i in range(base_image_channels):
        # build MPs
        mp_temp = build_morphological_profiles(
            base_image[:, :, i], se_size, se_size_increment, num_openings_closings)

        aux = morphological_profile_size * (i+1)

        # build the EMP
        cont_aux = 0
        for k in range(cont, aux):
            emp[:, :, k] = mp_temp[:, :, cont_aux]
            cont_aux += 1

        cont = morphological_profile_size * (i+1)
    return emp

dataset_names = ['IP', 'SA', 'PU']
parser = argparse.ArgumentParser(description="Run deep learning experiments on"
                                             " various hyperspectral datasets")
parser.add_argument('--dataset', type=str, default='PU', choices=dataset_names,
                    help="Dataset to use.")

parser.add_argument('--perclass', type=int, default=6)# 会除以100

args = parser.parse_args()

dataset = args.dataset
perclass=args.perclass

output_units = 9 if (dataset == 'PU' or dataset == 'PC') else 16
#IP 10249   PU 42776   SA  54129
if dataset=='IP':
    Total=10249
elif dataset=='PU':
    Total=42776
elif dataset=='SA':
    Total=54129
if perclass>1:
    perclass=int(perclass)
    test_ratio = Total-output_units*perclass
else:
    test_ratio =1-perclass
pixels, gt = loadData(dataset)

number_of_rows=pixels.shape[0]
number_of_columns=pixels.shape[1]
from sklearn.preprocessing import StandardScaler
pixels=np.reshape(pixels,(-1,pixels.shape[2]))


sc = StandardScaler()
pixels = sc.fit_transform(pixels)
from sklearn.decomposition import PCA
number_of_pc = 15
pca = PCA(n_components=number_of_pc)
pc = pca.fit_transform(pixels)
pc_images = np.zeros(shape=(number_of_rows, number_of_columns, number_of_pc))
for i in range(number_of_pc):
    pc_images[:, :, i] = np.reshape(pc[:, i], (number_of_rows, number_of_columns))


pc_images.shape
num_openings_closings = 4
morphological_profile_size = (num_openings_closings * 2) + 1
emp_image = build_emp(base_image=pc_images, num_openings_closings=num_openings_closings)
dim_x, dim_y, dim_z = emp_image.shape
dim = dim_x * dim_y
x = np.zeros(shape=(dim, dim_z))
y = np.reshape(gt,(-1))
cont = 0
for i in range(dim_x):
    for j in range(dim_y):
        x[cont, :] = emp_image[i, j, :]
        cont += 1

stratify=np.arange(0,output_units,1)
x_train, x_test, y_train, y_test = PerClassSplit(x, y, perclass, stratify,randomState=0)

# Fitting Kernel SVM to the Training set
from sklearn.svm import SVC
classifier = SVC(kernel='rbf', random_state=0)
classifier.fit(x_train, y_train)

# Predicting the Test set results
Y_pred = classifier.predict(x_test)




classification = classification_report(y_test, Y_pred)
test_end_time=time.time()
print(classification)

classification, confusion, oa, each_acc, aa, kappa,target_names = reports(Y_pred,y_test, dataset)

# each_acc_str = ','.join(str(x) for x in each_acc)
# add_info=[dataset,perclass,windowSize, oa,aa,kappa,train_end_time-train_start_time,test_end_time-test_start_time]+each_acc_str.split('[')[0].split(']')[0].split(',')
# csvFile = open("CombineExperiment2.csv", "a")
# writer = csv.writer(csvFile)
# writer.writerow(add_info)
# csvFile.close()

#
# def Patch(data, height_index, width_index):
#     height_slice = slice(height_index, height_index + PATCH_SIZE)
#     width_slice = slice(width_index, width_index + PATCH_SIZE)
#     patch = data[height_slice, width_slice, :]
#
#     return patch
#
#
# # load the original image
# X, y = loadData(dataset)
# height = y.shape[0]
# width = y.shape[1]
# PATCH_SIZE = windowSize
# numComponents = K
# X, pca = applyPCA(X, numComponents=numComponents)
# X = padWithZeros(X, PATCH_SIZE // 2)
# # calculate the predicted image
# outputs = np.zeros((height, width))
# for i in range(height):
#     for j in range(width):
#         target = int(y[i, j])
#         if target == 0:
#             continue
#         else:
#             image_patch = Patch(X, i, j)
#             X_test_image = image_patch.reshape(1,image_patch.shape[0], image_patch.shape[1],image_patch.shape[2]).astype('float32')
#             np.save('WholePic.npy',X_test_image)
#             Datapath='WholePic.npy'
#             Labelpath='WholePic.npy'
#             prediction = (predict(model,Datapath,Labelpath))
#             Prediction = np.argmax(np.array(prediction), axis=1)
#             outputs[i][j] = Prediction +1
# all_target=['untruthed']+target_names
# labelPatches = [ patches.Patch(color=spectral.spy_colors[x]/255.,
#                  label=all_target[x]) for x in np.unique(y) ]
#
# ground_truth = spectral.imshow(classes=y, figsize=(10, 10))
# plt.legend(handles=labelPatches, ncol=2, fontsize='medium',
#            loc='upper center', bbox_to_anchor=(0.5, -0.05));
# plt.savefig(str(dataset) + "_ground_truth.pdf", bbox_inches='tight',dpi=300)
#
# labelPatches2 = [ patches.Patch(color=spectral.spy_colors[x]/255.,
#                  label=all_target[x]) for x in np.unique(outputs.astype(int)) ]
# predict_image = spectral.imshow(classes=outputs.astype(int), figsize=(10, 10))
# plt.legend(handles=labelPatches2, ncol=2, fontsize='medium',
#            loc='upper center', bbox_to_anchor=(0.5, -0.05));
# plt.savefig(str(dataset) + "_predictions.pdf", bbox_inches='tight',dpi=300)
torch.cuda.empty_cache()