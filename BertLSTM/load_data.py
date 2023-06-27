"""
@Project : bert-enhancer-main (1) 
@File    : load_data.py
@Author  : endeavor
@Date    : 2023/4/1 22:07 
@Brief   : 
"""
import os
import torch
import pandas as pd
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split


class CSVDataset(Dataset):
    def __init__(self, root_dir, label):
        self.root_dir = root_dir
        self.files = os.listdir(root_dir)
        self.label = label

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        # 加载CSV文件
        file_path = os.path.join(self.root_dir, self.files[idx])
        data = pd.read_csv(file_path, header=None)
        # 将数据转换为PyTorch张量并返回
        data_tensor = torch.tensor(data.values, dtype=torch.float32)
        label_tensor = torch.tensor(self.label, dtype=torch.float32)
        return data_tensor, label_tensor


# 加载样本数据并合并
def LoadData(root_dir_pos, root_dir_neg):
    # 加载正样本数据
    csv_dataset_pos = CSVDataset(root_dir_pos, label=1)
    # 加载负样本数据
    csv_dataset_neg = CSVDataset(root_dir_neg, label=0)
    # 将正负数据合并到一起
    csv_dataset = torch.utils.data.ConcatDataset([csv_dataset_pos, csv_dataset_neg])
    return csv_dataset


def split_data():
    left_dataset = LoadData('../seq2/output_left_pos_csv', '../seq2/output_left_neg_csv')
    train_left_dataset, test_left_dataset = train_test_split(left_dataset, test_size=0.2, random_state=42)
    right_dataset = LoadData('../seq2/output_right_pos_csv', '../seq2/output_right_neg_csv')
    train_right_dataset, test_right_dataset = train_test_split(right_dataset, test_size=0.2, random_state=42)
    all_dataset = LoadData('../seq2/output_pos_csv', '../seq2/output_neg_csv')
    train_all_dataset, test_all_dataset = train_test_split(all_dataset, test_size=0.2, random_state=42)
    return train_left_dataset, train_right_dataset, train_all_dataset, test_left_dataset, test_right_dataset, test_all_dataset
