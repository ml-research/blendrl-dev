import torch as th

def true(objs: th.Tensor) -> th.Tensor:
    return th.ones(objs.size(0), device=objs.device)

def _oriented(obj: th.Tensor, values: th.Tensor) -> th.Tensor:
    orientations = obj[:, 3].int()
    mask = (orientations >= 0) & (orientations <= 15)
    x = th.zeros(obj.shape[0], device=obj.device)
    x[mask] = values[orientations[mask]]
    x[~mask] = 0.0

    return x

def left_oriented(obj: th.Tensor) -> th.Tensor:
    lvalues = th.linspace(0.99, 0.5, 8, device=obj.device)
    rvalues = 1 - lvalues.flip(dims=[0])
    values = th.concat((lvalues, rvalues), dim=-1)
    return _oriented(obj, values)

def right_oriented(obj: th.Tensor) -> th.Tensor:
    lvalues = th.linspace(0.01, 0.5, 8, device=obj.device)
    rvalues = 1 - lvalues.flip(dims=[0])
    values = th.concat((lvalues, rvalues), dim=-1)
    return _oriented(obj, values)

def straight_oriented(obj: th.Tensor) -> th.Tensor:
    lvalues = th.linspace(0.01, 0.99, 8, device=obj.device)
    rvalues = lvalues.flip(dims=[0])
    values = th.concat((lvalues, rvalues), dim=-1)
    return _oriented(obj, values)