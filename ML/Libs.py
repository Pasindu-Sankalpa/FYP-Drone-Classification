import os
import time
import copy
import librosa
import subprocess
import numpy as np
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

import torch
import torch.nn as nn
import torchvision
from torch.utils.data import Dataset, DataLoader, random_split

import warnings; warnings.filterwarnings('ignore')