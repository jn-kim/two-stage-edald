'''
This file contains code adapted from BalancedEntropy repository.
Source: https://github.com/jaeohwoo/BalancedEntropy/blob/main/src/acquisition_functions.py
'''

import enum
import torch
from .torch_utils import *

def bald_acquisition_function(log_p_B_K_C):
    return mutual_information(log_p_B_K_C)

def power_bald_acquisition_function(log_p_B_K_C):
    return power_bald(log_p_B_K_C)

def balentacq_acquisition_function(log_p_B_K_C):
    return balentacq(log_p_B_K_C)

class AcquisitionFunction(enum.Enum):
    bald       = "bald"
    power_bald = "power_bald"
    balentacq  = "balentacq"

    @property
    def scorer(self):
        if self is AcquisitionFunction.bald:
            return bald_acquisition_function
        elif self is AcquisitionFunction.power_bald:
            return power_bald_acquisition_function
        elif self is AcquisitionFunction.balentacq:
            return balentacq_acquisition_function
        else:
            raise NotImplementedError(f"{self} not supported yet!")

    def compute_scores(self, log_p_B_K_C, _, device):
        with torch.no_grad():
            return self.scorer(log_p_B_K_C.to(device)).double()
