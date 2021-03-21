from collections import Counter
import torch
from torchvision import models
from torch import nn
import pytorch_lightning as pl
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, random_split, Subset, Dataset
from torch.nn import functional as F
from torchvision import datasets, transforms
import torch.optim as optim
import os
from tqdm import tqdm
from pytorch_lightning.callbacks import ModelCheckpoint
from collections import Counter
import statistics

from torchvision.datasets.utils import download_and_extract_archive
import pandas as pd
import numpy as np
from sklearn import preprocessing

###------------------------------Utility function for DataLoader---------------------------------###
# Used to discard labels with occurence smaller than 10


def preprocess_data(full_labels, import_size=None):

    # Step #1: Convert from string into integer labels
    le = preprocessing.LabelEncoder()
    le.fit(full_labels.ravel())
    int_labels = le.transform(full_labels.ravel())

    # Step #2: Prepare indices for the proportion of data that we are going to read in
    full_indices = range(len(full_labels))
    if import_size == 1 or import_size is None:
        import_indices = full_indices
        discarded_indices = []
    else:
        import_indices, discarded_indices = train_test_split(
            full_indices, train_size=import_size, stratify=int_labels, random_state=21)

    print("Number of data to import:", len(import_indices))
    print("Number of total data:", len(full_labels))

    # Step #3: Preprocess data and only keep cells with population larger than 10
    imported_labels = [int_labels[i] for i in import_indices]
    occurence_dict = Counter(imported_labels)
    remove_labels = []
    for label in occurence_dict:
        num_of_occurence = occurence_dict[label]
        if num_of_occurence < 10:
            remove_labels.append(label)
    imported_labels = None
    import_indices = None

    for label in remove_labels:
        remove_temp = [i for i, x in enumerate(int_labels) if x == label]
        discarded_indices.extend(remove_temp)

    remaining_indices = [e for e in full_indices if e not in discarded_indices]
    remaining_labels = [int_labels[i] for i in remaining_indices]
    print("Number of data after filtering:", len(remaining_indices))
    print("Number of classes after filtering:",
          len(np.unique(remaining_labels)))

    remaining_labels = list(le.inverse_transform(remaining_labels))

    return remaining_labels, discarded_indices

# Perform stratified split on a dataset into two sets based on indices


def stratified_split(remaining_indices, full_labels, set1_split_percentage):
    target_labels = [full_labels[i] for i in remaining_indices]
    set1_indices, set2_indices = train_test_split(
        remaining_indices, train_size=set1_split_percentage, stratify=target_labels)
    return set1_indices, set2_indices

# Split a full datasets in a stratified way into test, train, validation and database sets


def split_test_train_val_database_sets(full_dataset, train_percentage, val_percentage, test_percentage):
    dataset_percentage = 1 - train_percentage - val_percentage - test_percentage
    full_labels = full_dataset.labels
    full_indices = range(len(full_labels))

    database_indices, remaining_indices = stratified_split(
        full_indices, full_labels, dataset_percentage)
    train_indices, remaining_indices = stratified_split(
        remaining_indices, full_labels, train_percentage/(1 - dataset_percentage))
    val_indices, test_indices = stratified_split(
        remaining_indices, full_labels, val_percentage/(test_percentage + val_percentage))

    TM_database, TM_train, TM_val, TM_test = (Subset(full_dataset, database_indices),
                                              Subset(full_dataset,
                                                     train_indices),
                                              Subset(full_dataset,
                                                     val_indices),
                                              Subset(full_dataset, test_indices))
    return TM_database, TM_train, TM_val, TM_test


class CustomDataset(Dataset):
    'A dataset base class for PyTorch Lightening'

    def __init__(self, data, labels):
        'Dataset Class Initialization'
        # Number of data and labels should match
        assert len(data) == len(labels)
        self.labels = labels
        self.data = data

    def __len__(self):
        'Returns the total number of samples'
        return len(self.data)

    def __getitem__(self, index: int):
        # Load data and get label
        return self.data[index], self.labels[index]


###------------------------------Data Module---------------------------------###

class TMDataModule(pl.LightningDataModule):

    def __init__(self, data_dir: str = './data', batch_size=64, num_workers=2, import_size=0.4):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.data_name = "TM"
        # Percentage of original dataset, Total dataset size = 54865. Went to 0.4 without crashing
        self.import_size = import_size
        self.label_mapping = None
        self.N_FEATURES = 19791
        self.N_CLASS = 55

    def prepare_data(self):
        # download
        if not os.path.exists(f"{self.data_dir}/{self.data_name}"):
            print("Start downloading TM!")
            url = 'https://github.com/Aprilhuu/Deep-Learning-in-Single-Cell-Analysis/raw/main/TM.zip'
            download_and_extract_archive(
                url, f"{self.data_dir}", f"{self.data_dir}", filename=self.data_name + '.zip')

    def setup(self, stage):
        # Assign train/val datasets for use in dataloaders

        DataPath = self.data_dir + "/" + self.data_name + "/Filtered_TM_data.csv"
        LabelsPath = self.data_dir + "/" + self.data_name + "/Labels.csv"

        # Step #1: Read in all labels
        labels = pd.read_csv(LabelsPath, header=0, index_col=None, sep=',')
        full_labels = np.asarray(labels)
        labels = None

        # Step #2: Preprocess data and only keep cells with population larger than 10
        remaining_labels, discarded_indices = preprocess_data(
            full_labels, self.import_size)

        # Step #3: Turning string class labels into int labels and store the mapping
        self.label_mapping = preprocessing.LabelEncoder()
        self.label_mapping.fit(remaining_labels)
        int_labels = self.label_mapping.transform(remaining_labels)

        full_labels = np.asarray(int_labels)
        remaining_labels = None
        int_labels = None

        # Step #4: Read in data based on selected label indices
        discarded_indices = [x + 1 for x in discarded_indices]
        data = pd.read_csv(DataPath, index_col=0, sep=',',
                           skiprows=discarded_indices)
        discarded_indices = None
        full_data = np.asarray(data, dtype=np.float32)
        full_dataset = CustomDataset(full_data, full_labels)

        # Step #5: Split indices in stratified way into train, validation, test and database sets
        #          and prepare all datasets based on splited list indices
        self.TM_database, self.TM_train, self.TM_val, self.TM_test = split_test_train_val_database_sets(full_dataset,
                                                                                                        train_percentage=0.3,
                                                                                                        val_percentage=0.1,
                                                                                                        test_percentage=0.1)

        self.database_dataloader = DataLoader(self.TM_database, batch_size=self.batch_size,
                                              num_workers=self.num_workers)

        print("database size =", len(self.TM_database))
        print("train size =", len(self.TM_train))
        print("val size =", len(self.TM_val))
        print("test size =", len(self.TM_test))

        # Calculate sample count in each class for training dataset
        samples_in_each_class_dict = Counter(
            [data[1] for data in self.TM_train])
        print("training samples in each class =",
              sorted(samples_in_each_class_dict.items()))
        samples_in_each_class_dict_val = Counter(
            [data[1] for data in self.TM_val])
        print("val samples in each class =", sorted(
            samples_in_each_class_dict_val.items()))
        samples_in_each_class_dict_test = Counter(
            [data[1] for data in self.TM_test])
        print("test samples in each class =", sorted(
            samples_in_each_class_dict_test.items()))
        self.N_CLASS = len(samples_in_each_class_dict)
        print("Changing N_CLASS =", self.N_CLASS)
        self.samples_in_each_class = torch.zeros(self.N_CLASS)
        for index, count in samples_in_each_class_dict.items():
            self.samples_in_each_class[index] = count

    def train_dataloader(self):
        return DataLoader(self.TM_train, batch_size=self.batch_size,
                          shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.TM_val, batch_size=self.batch_size,
                          num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.TM_test, batch_size=self.batch_size,
                          num_workers=self.num_workers)


class BaronHumanDataModule(pl.LightningDataModule):

    def __init__(self, data_dir: str = './data', batch_size=64, num_workers=2):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.data_name = "BaronHuman"
        self.label_mapping = None

    def prepare_data(self):
        # download
        if not os.path.exists(f"{self.data_dir}/{self.data_name}"):
            print("Start downloading Baron Human!")
            url = 'https://github.com/Aprilhuu/Deep-Learning-in-Single-Cell-Analysis/raw/main/BaronHuman.zip'
            download_and_extract_archive(
                url, f"{self.data_dir}", f"{self.data_dir}", filename=self.data_name + '.zip')

    def setup(self, stage):
        # Assign train/val datasets for use in dataloaders

        DataPath = self.data_dir + "/" + self.data_name + \
            "/Filtered_Baron_HumanPancreas_data.csv"
        LabelsPath = self.data_dir + "/" + self.data_name + "/Labels.csv"

        # Step #1: Read in all labels
        labels = pd.read_csv(LabelsPath, header=0, index_col=None, sep=',')
        full_labels = np.asarray(labels)
        labels = None

        # Step #2: Preprocess data and only keep cells with population larger than 10
        remaining_labels, discarded_indices = preprocess_data(full_labels)

        # Step #3: Turning string class labels into int labels and store the mapping
        self.label_mapping = preprocessing.LabelEncoder()
        self.label_mapping.fit(remaining_labels)
        int_labels = self.label_mapping.transform(remaining_labels)

        full_labels = np.asarray(int_labels)
        remaining_labels = None
        int_labels = None

        # Step #4: Read in data based on selected label indices
        discarded_indices = [x + 1 for x in discarded_indices]
        data = pd.read_csv(DataPath, index_col=0, sep=',',
                           skiprows=discarded_indices)
        discarded_indices = None
        full_data = np.asarray(data, dtype=np.float32)
        full_dataset = CustomDataset(full_data, full_labels)

        self.Baron_Human_database, self.Baron_Human_train, self.Baron_Human_val, self.Baron_Human_test = split_test_train_val_database_sets(full_dataset,
                                                                                                                                            train_percentage=0.3,
                                                                                                                                            val_percentage=0.1,
                                                                                                                                            test_percentage=0.1)

        self.database_dataloader = DataLoader(self.Baron_Human_database, batch_size=self.batch_size,
                                              num_workers=self.num_workers)

        print("database size =", len(self.Baron_Human_database))
        print("train size =", len(self.Baron_Human_train))
        print("val size =", len(self.Baron_Human_val))
        print("test size =", len(self.Baron_Human_test))

        # Calculate sample count in each class for training dataset
        samples_in_each_class_dict = Counter(
            [data[1] for data in self.Baron_Human_train])
        print("training samples in each class =", samples_in_each_class_dict)
        samples_in_each_class_dict_val = Counter(
            [data[1] for data in self.Baron_Human_val])
        print("val samples in each class =", samples_in_each_class_dict_val)
        samples_in_each_class_dict_test = Counter(
            [data[1] for data in self.Baron_Human_test])
        print("test samples in each class =", samples_in_each_class_dict_test)
        self.N_CLASS = len(samples_in_each_class_dict)
        print("Changing N_CLASS =", self.N_CLASS)
        self.samples_in_each_class = torch.zeros(self.N_CLASS)
        for index, count in samples_in_each_class_dict.items():
            self.samples_in_each_class[index] = count

    def train_dataloader(self):
        return DataLoader(self.Baron_Human_train, batch_size=self.batch_size,
                          shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.Baron_Human_val, batch_size=self.batch_size,
                          num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.Baron_Human_test, batch_size=self.batch_size,
                          num_workers=self.num_workers)


class Zheng68KDataModule(pl.LightningDataModule):

    def __init__(self, data_dir: str = './data', batch_size=64, num_workers=2, import_size=0.4):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.data_name = "Zheng_68K"
        self.import_size = import_size # Percentage of original dataset, Total dataset size = 54865. Went to 0.4 without crashing
        self.label_mapping = None
        self.N_FEATURES = 20387
        self.N_CLASS = 11

    def prepare_data(self):
        # download
        if not os.path.exists(f"{self.data_dir}/{self.data_name}"):
          print("Start downloading Zheng 68K!")
          url = 'https://github.com/Aprilhuu/Deep-Learning-in-Single-Cell-Analysis/raw/main/Zheng_68K.zip'
          download_and_extract_archive(url, f"{self.data_dir}", f"{self.data_dir}", filename=self.data_name + '.zip')

    def setup(self, stage):
        # Assign train/val datasets for use in dataloaders

        DataPath = self.data_dir + "/" + self.data_name + "/Filtered_68K_PBMC_data.csv"
        LabelsPath = self.data_dir + "/" + self.data_name + "/Labels.csv"
        
        # Step #1: Read in all labels
        labels = pd.read_csv(LabelsPath, header=0, index_col=None, sep=',')
        full_labels = np.asarray(labels)
        labels = None

        # Step #2: Preprocess data and only keep cells with population larger than 10
        remaining_labels, discarded_indices = preprocess_data(full_labels, self.import_size)

        # Step #3: Turning string class labels into int labels and store the mapping
        self.label_mapping = preprocessing.LabelEncoder()
        self.label_mapping.fit(remaining_labels)
        int_labels = self.label_mapping.transform(remaining_labels)

        full_labels = np.asarray(int_labels)
        remaining_labels = None
        int_labels = None

        # Step #4: Read in data based on selected label indices
        discarded_indices = [x + 1 for x in discarded_indices]
        data = pd.read_csv(DataPath, index_col=0, sep=',', skiprows=discarded_indices)
        discarded_indices = None
        full_data = np.asarray(data, dtype=np.float32)
        full_dataset = CustomDataset(full_data, full_labels)

        # Step #5: Split indices in stratified way into train, validation, test and database sets 
        #          and prepare all datasets based on splited list indices
        self.Zheng_68K_database, self.Zheng_68K_train, self.Zheng_68K_val, self.Zheng_68K_test = split_test_train_val_database_sets(full_dataset, 
                                                                                                        train_percentage=0.3,
                                                                                                        val_percentage=0.1, 
                                                                                                        test_percentage=0.1)
        
        self.database_dataloader = DataLoader(self.Zheng_68K_database, batch_size=self.batch_size,
                                              num_workers=self.num_workers)
         
        print("database size =", len(self.Zheng_68K_database))
        print("train size =", len(self.Zheng_68K_train))
        print("val size =", len(self.Zheng_68K_val))
        print("test size =", len(self.Zheng_68K_test))
        
        # Calculate sample count in each class for training dataset
        samples_in_each_class_dict = Counter([data[1] for data in self.Zheng_68K_train])
        print("training samples in each class =", sorted(samples_in_each_class_dict.items()))
        samples_in_each_class_dict_val = Counter([data[1] for data in self.Zheng_68K_val])
        print("val samples in each class =", sorted(samples_in_each_class_dict_val.items()))
        samples_in_each_class_dict_test = Counter([data[1] for data in self.Zheng_68K_test])
        print("test samples in each class =", sorted(samples_in_each_class_dict_test.items()))
        self.N_CLASS = len(samples_in_each_class_dict)
        print("Changing N_CLASS =", self.N_CLASS)
        self.samples_in_each_class = torch.zeros(self.N_CLASS)
        for index, count in samples_in_each_class_dict.items():
           self.samples_in_each_class[index] = count


    def train_dataloader(self):
        return DataLoader(self.Zheng_68K_train, batch_size=self.batch_size,
                          shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.Zheng_68K_val, batch_size=self.batch_size,
                          num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.Zheng_68K_test, batch_size=self.batch_size,
                          num_workers=self.num_workers)


class AMBDataModule(pl.LightningDataModule):

    def __init__(self, data_dir: str = './data', batch_size=64, num_workers=2, annotation_level=92):
        super().__init__()
        assert annotation_level in [3, 16, 92], "Annotation level must be one of 3, 16 or 92!"
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.data_name = "AMB"
        self.annotation_level = annotation_level
        self.label_mapping = None

    def prepare_data(self):
        # download
        if not os.path.exists(f"{self.data_dir}/{self.data_name}"):
          print("Start downloading AMB!")
          url = 'https://github.com/Aprilhuu/Deep-Learning-in-Single-Cell-Analysis/raw/main/AMB.zip'
          download_and_extract_archive(url, f"{self.data_dir}", f"{self.data_dir}", filename=self.data_name + '.zip')

    def setup(self, stage):
        # Assign train/val datasets for use in dataloaders

        DataPath = self.data_dir + "/" + self.data_name + "/Filtered_mouse_allen_brain_data.csv"
        LabelsPath = self.data_dir + "/" + self.data_name + "/Labels.csv"

        # Step #1: Read in all labels
        if self.annotation_level == 92:
          labels = pd.read_csv(LabelsPath, header=0, index_col=None, sep=',', usecols = ['cluster'])
        elif self.annotation_level == 16:
          labels = pd.read_csv(LabelsPath, header=0, index_col=None, sep=',', usecols = ['Subclass'])
        else:
          labels = pd.read_csv(LabelsPath, header=0, index_col=None, sep=',', usecols = ['Class'])
          
        full_labels = np.asarray(labels)
        labels = None

        # Step #2: Preprocess data and only keep cells with population larger than 10
        remaining_labels, discarded_indices = preprocess_data(full_labels)

        # Step #3: Turning string class labels into int labels and store the mapping
        self.label_mapping = preprocessing.LabelEncoder()
        self.label_mapping.fit(remaining_labels)
        int_labels = self.label_mapping.transform(remaining_labels)

        full_labels = np.asarray(int_labels)
        remaining_labels = None
        int_labels = None

        # Step #4: Read in data based on selected label indices
        discarded_indices = [x + 1 for x in discarded_indices]
        data = pd.read_csv(DataPath, index_col=0, sep=',', skiprows=discarded_indices)
        discarded_indices = None
        full_data = np.asarray(data, dtype=np.float32)
        full_dataset = CustomDataset(full_data, full_labels)

        label_mapping = dict()
        for i in range(len(self.label_mapping.classes_)):
          label_mapping[i] = self.label_mapping.classes_[i]

        print("Label name to integer mappings:", label_mapping)

        for i in range (5):
          print(full_dataset[i])
        
        self.AMB_database, self.AMB_train, self.AMB_val, self.AMB_test = split_test_train_val_database_sets(full_dataset, 
                                                                   train_percentage=0.3,
                                                                   val_percentage=0.1, 
                                                                   test_percentage=0.1)

        self.database_dataloader = DataLoader(self.AMB_database, batch_size=self.batch_size,
                             num_workers=self.num_workers)
        
        print("database size =", len(self.AMB_database))
        print("train size =", len(self.AMB_train))
        print("val size =", len(self.AMB_val))
        print("test size =", len(self.AMB_test))

        # Calculate sample count in each class for training dataset
        samples_in_each_class_dict = Counter([data[1] for data in self.AMB_train])
        print("training samples in each class =", samples_in_each_class_dict)
        samples_in_each_class_dict_val = Counter([data[1] for data in self.AMB_val])
        print("val samples in each class =", samples_in_each_class_dict_val)
        samples_in_each_class_dict_test = Counter([data[1] for data in self.AMB_test])
        print("test samples in each class =", samples_in_each_class_dict_test)
        self.N_CLASS = len(samples_in_each_class_dict)
        print("Changing N_CLASS =", self.N_CLASS)
        self.samples_in_each_class = torch.zeros(self.N_CLASS)
        for index, count in samples_in_each_class_dict.items():
           self.samples_in_each_class[index] = count

    def train_dataloader(self):
        return DataLoader(self.AMB_train, batch_size=self.batch_size,
                          shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.AMB_val, batch_size=self.batch_size,
                          num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.AMB_test, batch_size=self.batch_size,
                          num_workers=self.num_workers)


class XinDataModule(pl.LightningDataModule):

    def __init__(self, data_dir: str = './data', batch_size=64, num_workers=2):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.data_name = "Xin"
        self.label_mapping = None

    def prepare_data(self):
        # download
        if not os.path.exists(f"{self.data_dir}/{self.data_name}"):
          print("Start downloading Xin!")
          url = 'https://github.com/Aprilhuu/Deep-Learning-in-Single-Cell-Analysis/raw/main/Xin.zip'
          download_and_extract_archive(url, f"{self.data_dir}", f"{self.data_dir}", filename=self.data_name + '.zip')

    def setup(self, stage):
        # Assign train/val datasets for use in dataloaders

        DataPath = self.data_dir + "/" + self.data_name + "/Filtered_Xin_HumanPancreas_data.csv"
        LabelsPath = self.data_dir + "/" + self.data_name + "/Labels.csv"

        # Step #1: Read in all labels
        labels = pd.read_csv(LabelsPath, header=0, index_col=None, sep=',')
        full_labels = np.asarray(labels)
        labels = None

        # Step #2: Preprocess data and only keep cells with population larger than 10
        remaining_labels, discarded_indices = preprocess_data(full_labels)

        # Step #3: Turning string class labels into int labels and store the mapping
        self.label_mapping = preprocessing.LabelEncoder()
        self.label_mapping.fit(remaining_labels)
        int_labels = self.label_mapping.transform(remaining_labels)

        full_labels = np.asarray(int_labels)
        remaining_labels = None
        int_labels = None

        # Step #4: Read in data based on selected label indices
        discarded_indices = [x + 1 for x in discarded_indices]
        data = pd.read_csv(DataPath, index_col=0, sep=',', skiprows=discarded_indices)
        discarded_indices = None
        full_data = np.asarray(data, dtype=np.float32)
        full_dataset = CustomDataset(full_data, full_labels)
        
        self.Xin_database, self.Xin_train, self.Xin_val, self.Xin_test = split_test_train_val_database_sets(full_dataset, 
                                                                   train_percentage=0.3,
                                                                   val_percentage=0.1, 
                                                                   test_percentage=0.1)

        self.database_dataloader = DataLoader(self.Xin_database, batch_size=self.batch_size,
                             num_workers=self.num_workers)
        
        print("database size =", len(self.Xin_database))
        print("train size =", len(self.Xin_train))
        print("val size =", len(self.Xin_val))
        print("test size =", len(self.Xin_test))

        # Calculate sample count in each class for training dataset
        samples_in_each_class_dict = Counter([data[1] for data in self.Xin_train])
        print("training samples in each class =", samples_in_each_class_dict)
        samples_in_each_class_dict_val = Counter([data[1] for data in self.Xin_val])
        print("val samples in each class =", samples_in_each_class_dict_val)
        samples_in_each_class_dict_test = Counter([data[1] for data in self.Xin_test])
        print("test samples in each class =", samples_in_each_class_dict_test)
        self.N_CLASS = len(samples_in_each_class_dict)
        print("Changing N_CLASS =", self.N_CLASS)
        self.samples_in_each_class = torch.zeros(self.N_CLASS)
        for index, count in samples_in_each_class_dict.items():
           self.samples_in_each_class[index] = count

    def train_dataloader(self):
        return DataLoader(self.Xin_train, batch_size=self.batch_size,
                          shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.Xin_val, batch_size=self.batch_size,
                          num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.Xin_test, batch_size=self.batch_size,
                          num_workers=self.num_workers)
