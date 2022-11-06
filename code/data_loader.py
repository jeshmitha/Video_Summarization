import torch
from torch.utils.data import Dataset, DataLoader
import h5py
import numpy as np
import json

class VideoData(Dataset):
    def __init__(self, setting, mode, split_index):
        self.setting = setting
        self.mode = mode
        self.splits_filename = ['./data/splits/' + self.setting + '_splits.json']
        self.splits = []
        self.split_index = split_index # it represents the current split (varies from 0 to 4)
        temp = {}

        self.summe_video_data = h5py.File('./data/SumMe/eccv16_dataset_summe_google_pool5.h5', 'r')
        self.tvsum_video_data = h5py.File('./data/TVSum/eccv16_dataset_tvsum_google_pool5.h5', 'r')
        self.ovp_video_data = h5py.File('./data/OVP/eccv16_dataset_ovp_google_pool5.h5', 'r')
        self.youtube_video_data = h5py.File('./data/YouTube/eccv16_dataset_youtube_google_pool5.h5', 'r')

        with open(self.splits_filename[0]) as f:
            data = json.loads(f.read())
            for split in data:
                temp['train_keys'] = split['train_keys']
                temp['test_keys'] = split['test_keys']
                self.splits.append(temp.copy())

    def __len__(self):
        self.len = len(self.splits[0][self.mode+'_keys'])
        return self.len

    # In "train" mode it returns the features; in "test" mode it also returns the video_name
    def __getitem__(self, index):
        video_key= self.splits[self.split_index][self.mode + '_keys'][index]
        video_name=video_key[2:]
        if video_key[0]=='t':
            frame_features = torch.Tensor(np.array(self.tvsum_video_data[video_name + '/features']))
        elif video_key[0]=='s':
            frame_features = torch.Tensor(np.array(self.summe_video_data[video_name + '/features']))
        elif video_key[0]=='o':
            frame_features = torch.Tensor(np.array(self.ovp_video_data[video_name + '/features']))
        elif video_key[0]=='y':
            frame_features = torch.Tensor(np.array(self.youtube_video_data[video_name + '/features']))
        return frame_features, video_key


def get_loader(setting, mode, split_index):
    return VideoData(setting, mode, split_index)