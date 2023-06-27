"""
@Project : Multi-LSTM (1)
@File    : save.py
@Author  : endeavor
@Brief   : The file is used for saving the best epoch results
"""
import torch


def save_checkpoint(state, is_best, OutputDir, test_index):
    if is_best:
        print('=> Saving a new best from epoch %d"' % state['epoch'])
        torch.save(state, OutputDir + '/' + str(test_index+1) + '_checkpoint.pth.tar')

    else:
        print("=> Validation Performance did not improve")
