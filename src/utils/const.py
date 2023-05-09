"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.

constants
"""
IMG_DIM = 2048
IMG_LABEL_DIM = 1601
BUCKET_SIZE = 8192
loss_methods = {'eg': 0, 'vg': 1, 'mg': 2, 'none': 3}
loss_methods_v2k = {v:k for k,v in loss_methods.items()}
loss_methods_nums = 4
