from .expert import UpstreamExpert as _UpstreamExpert


def data2vec_hug(ckpt, *args, **kwargs):
    """
        ckpt:
            The identifier string of huggingface data2vec models.
            eg. facebook/data2vec-audio-base
            see https://huggingface.co/facebook
    """

    return _UpstreamExpert(ckpt, *args, **kwargs)



def data2vec_hug_base(*args, **kwargs):
    kwargs['ckpt'] = 'facebook/data2vec-audio-base'
    return data2vec_hug(*args, **kwargs)


def data2vec_hug_large(*args, **kwargs):
    kwargs['ckpt'] = 'facebook/data2vec-audio-large'
    return data2vec_hug(*args, **kwargs)