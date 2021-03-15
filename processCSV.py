import csv
import pandas as pd

data = pd.read_csv(r'ActiveLearning2.csv')
csvFile = open("MeanActive.csv", "a")
writer = csv.writer(csvFile)
add_info=data.columns.str.strip(',')
writer.writerow(add_info)
csvFile.close()
for dataname in ['SA','PU','IP']:
    FixDataset=data.loc[data.dataset==dataname]
    for perclass in [3,6,9,12,15,18]:
        FixTem=FixDataset.loc[FixDataset.perclass==perclass]
        for Windowsize in [15]:
            FixWindow=FixTem.loc[FixTem.windowsize==Windowsize]

            add_info =FixWindow[FixWindow.columns[0:2]].values[0].tolist()+FixWindow[FixWindow.columns[2:]].values.mean(axis=0).tolist()
            csvFile = open("MeanActive.csv", "a")
            writer = csv.writer(csvFile)
            writer.writerow(add_info)
            csvFile.close()
            print(FixWindow)

data = pd.read_csv(r'CombineExperiment2.csv')
csvFile = open("MeanCombine.csv", "a")
writer = csv.writer(csvFile)
add_info=data.columns.str.strip(',')
writer.writerow(add_info)
csvFile.close()
for dataname in ['SA','PU','IP']:
    FixDataset=data.loc[data.dataset==dataname]
    for perclass in [3,6,9,12,15,18]:
        FixTem=FixDataset.loc[FixDataset.perclass==perclass]
        for Windowsize in [15]:
            FixWindow=FixTem.loc[FixTem.windowsize==Windowsize]

            add_info =FixWindow[FixWindow.columns[0:2]].values[0].tolist()+FixWindow[FixWindow.columns[2:]].values.mean(axis=0).tolist()
            csvFile = open("MeanCombine.csv", "a")
            writer = csv.writer(csvFile)
            writer.writerow(add_info)
            csvFile.close()
            print(FixWindow)
