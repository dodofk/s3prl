# Copyright (c) Facebook, Inc. All Rights Reserved

# -*- coding: utf-8 -*- #
"""*********************************************************************************************"""
#   FileName     [ upstream/data2vec/hubconf.py ]
#   Synopsis     [ the data2vec torch hubconf ]
#   Author       [Dodofk]
"""*********************************************************************************************"""


###############
# IMPORTATION #
###############
import os
#-------------#
from s3prl.utility.download import _urls_to_filepaths
from .expert import UpstreamExpert as _UpstreamExpert


def data2vec_local(ckpt, *args, **kwargs):
    """
        The model from local ckpt
            ckpt (str): PATH
    """
    assert os.path.isfile(ckpt)
    return _UpstreamExpert(ckpt, *args, **kwargs)


def data2vec_url(ckpt, refresh=False, *args, **kwargs):
    """
        The model from google drive id
            ckpt (str): URL
            refresh (bool): whether to download ckpt/config again if existed
    """
    return data2vec_local(_urls_to_filepaths(ckpt, refresh=refresh), *args, **kwargs)


def data2vec(refresh=False, *args, **kwargs):
    """
        The default model - Base
            refresh (bool): whether to download ckpt/config again if existed
    """
    return data2vec_base_960h(refresh=refresh, *args, **kwargs)


def data2vec_base_960h(refresh=False, *args, **kwargs):
    """
        The Base model
            refresh (bool): whether to download ckpt/config again if existed
    """
    kwargs['ckpt'] = 'https://dl.fbaipublicfiles.com/fairseq/data2vec/audio_base_ls.pt'
    return data2vec_url(refresh=refresh, *args, **kwargs)


def data2vec_large(refresh=False, *args, **kwargs):
    """
        The Large model
            refresh (bool): whether to download ckpt/config again if existed
    """
    kwargs['ckpt'] = 'https://dl.fbaipublicfiles.com/fairseq/data2vec/vox_pretrained.pt'
    return data2vec_url(refresh=refresh, *args, **kwargs)
