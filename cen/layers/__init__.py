import omegaconf

from . import contextual_dense
from . import contextual_mixture


def get_contextual(name, **kwargs):
    for key, value in kwargs.items():
        if isinstance(value, omegaconf.dictconfig.DictConfig):
            kwargs[key] = dict(value)
    if name == "affine":
        return contextual_dense.ContextualAffineDense(**kwargs)
    elif name == "convex":
        return contextual_dense.ContextualConvexDense(**kwargs)
    elif name == "mixture":
        return contextual_mixture.ContextualMixture(**kwargs)
    else:
        raise ValueError(f"Unknown contextual layer: {name}")
