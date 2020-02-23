from . import baseline
from . import cen
from . import moe


def get(name, **kwargs):
    if name == "baseline":
        build_model = baseline.build_model
    elif name == "cen":
        build_model = cen.build_model
    elif name == "moe":
        build_model = moe.build_model
    else:
        raise ValueError(f"Unknown model name: {name}.")

    return build_model(**kwargs)
