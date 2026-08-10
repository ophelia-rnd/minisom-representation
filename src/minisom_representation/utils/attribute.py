import inspect

def extract_attributes(cls, skeleton, **overrides):
    attributes = inspect.signature(cls).parameters
    kwargs = {
        p: getattr(skeleton, p, param.default)
        for p, param in attributes.items()
        if hasattr(skeleton, p) or param.default != inspect.Parameter.empty
    }
    kwargs.update(overrides)
    return kwargs
