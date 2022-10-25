import torch
import librosa
import librosa.util as librosa_util
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as  np
from pathlib import Path
from matplotlib import pyplot as plt

# 下面这个类可以做短时傅里叶变换(STFT)音频特征提取
class STFT(torch.nn.Module):
    """adapted from Prem Seetharaman's https://github.com/pseeth/pytorch-stft"""
    def __init__(self, filter_length=800, hop_length=200, win_length=800,
                 window='hann'):
        super(STFT, self).__init__()
        self.filter_length = filter_length
        self.hop_length = hop_length
        self.win_length = win_length
        self.window = window
        self.forward_transform = None
        
    def transform(self, input_data):
        num_batches = input_data.size(0)
        num_samples = input_data.size(1)
