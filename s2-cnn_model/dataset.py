import numpy as np
import pandas as pd
from collections import defaultdict
from os.path import join as pjoin
import torch
import torch.utils.data as tdata
from xz import RedisCached, info
import os


def prepare_folds(config):
    os.makedirs(config['pwd'], exist_ok=True)
    n_folds = config['n_folds']

    info(f"Fold cache does not exist. Creating folds anew and save the samples to {config['pwd']}.")
    pmeta = pd.read_csv(config['path_to_pmeta'])
    pmeta = pmeta.loc[pmeta['group'] == 'train']

    pmeta['i_fold'] = np.random.randint(0, n_folds, len(pmeta))

    for i_fold in range(n_folds):
        base_dir = pjoin(config['pwd'], f"fold_{i_fold}")
        os.makedirs(base_dir, exist_ok=True)
        path_to_valid = pjoin(base_dir, "valid_pmeta.csv")
        path_to_train = pjoin(base_dir, "train_pmeta.csv")
        pmeta.loc[pmeta['i_fold'] == i_fold].to_csv(path_to_valid)
        pmeta.loc[pmeta['i_fold'] != i_fold].to_csv(path_to_train)


def get_loader_pair(config):
    os.makedirs(config['pwd'], exist_ok=True)

    path_to_valid = f"{config['pwd']}/valid_pmeta.csv"
    path_to_train = f"{config['pwd']}/train_pmeta.csv"

    # if the cached split files exist, load them
    # TIP: if you do not want to load the cached split, simply delete them
    try:
        valid_pmeta = pd.read_csv(path_to_valid)
        train_pmeta = pd.read_csv(path_to_train)
        info(f"{path_to_valid} and {path_to_train} are loaded.")
    except FileNotFoundError:
        info(f"Split cache does not exist. Creating new split and save the samples to {path_to_valid} and {path_to_train}.")
        pmeta = pd.read_csv(config['path_to_pmeta'])
        pmeta = pmeta.loc[pmeta['group'] == 'train']

        n_samples = len(pmeta)
        n_samples_valid = round(config['pct_valid'] * n_samples)

        indices = torch.randperm(len(pmeta)).tolist()

        valid_pmeta = pmeta.iloc[indices[:n_samples_valid]]
        train_pmeta = pmeta.iloc[indices[n_samples_valid:]]

        valid_pmeta.to_csv(path_to_valid, index=False)
        train_pmeta.to_csv(path_to_train, index=False)

    valid_dataset = MyDataset(valid_pmeta, config, debug=False)
    train_dataset = MyDataset(train_pmeta, config, debug=False)

    train_loader = tdata.DataLoader(
        train_dataset,
        batch_size=config['train_batch_size'],
        shuffle=True,
        num_workers=70,
        collate_fn=train_dataset.collate_fn,
    )

    valid_loader = tdata.DataLoader(
        valid_dataset,
        batch_size=config['train_batch_size'],
        shuffle=False,
        num_workers=70,
        collate_fn=valid_dataset.collate_fn,
    )

    return train_loader, valid_loader


@RedisCached
def load_image(path_to_npy):
    layer = np.load(path_to_npy)
    return layer


class MyDataset(tdata.Dataset):

    def __init__(self, pmeta, config, debug=False):
        self.pmeta = pmeta
        self.debug = debug
        self.config = config

    def load_and_process_image(self, dicom_id):
        img = load_image(f"{self.config['dir_to_npys']}/{dicom_id}.npy")
        if self.config['single_channel']:
            img = img.reshape((1, img.shape[0], img.shape[1]))
        img = img.astype(np.float32)

        # TODO: Augmentation happens here
        if self.debug:
            return img.transpose((1, 2, 0))
        else:
            return img

    def __getitem__(self, index):
        row = self.pmeta.iloc[index]

        data_list = []
        for view in ['LCC', 'RCC', 'LMLO', 'RMLO']:
            data_list.append(self.load_and_process_image(row[view]))

        if self.debug:
            data = data_list
            label = f"""\
index = {index}
LCC_filename = {row['LCC']}
group = {row['group']}
label = {row['label']}
"""
        else:
            data = np.array(data_list)
            if self.config['phase'] == 'p1':
                label = np.array([0 if row['label'] == 0 else 1], dtype=np.float32)
            elif self.config['phase'] == 'p2':
                label = (np.arange(3) == row['label']).astype(np.float32)
            else:
                raise ValueError()

        return data, label

    def __len__(self):
        return len(self.pmeta)

    def collate_fn(self, batch):
        if self.debug:
            return batch

        data_list = []
        label_list = []
        for data, label in batch:
            data_list.append(data)
            label_list.append(label)

        return torch.tensor(data_list).transpose(0, 1), torch.tensor(label_list)
