from collections.abc import Callable

from torch import Tensor


def apply_to_dict(func: Callable, tensor_dict: dict[str, Tensor] | list[dict[str, Tensor]], **kwargs):
    """Applies func to each value of tensor_dict(s).
    The keys of the output dictionary are combined from the first argument's key and the result's key.
    """
    out = {}
    if isinstance(tensor_dict, dict):
        tensor_dict = [tensor_dict]

    for key_value_pairs in zip(*[d.items() for d in tensor_dict]):
        keys = [kv[0] for kv in key_value_pairs]
        values = [kv[1] for kv in key_value_pairs]

        prefix = keys[0].split("_")[0]
        if not all([key.startswith(prefix) for key in keys]):
            raise ValueError("Prefixes of tensor_dicts must match.")

        for result_key, result in func(*values, **kwargs).items():
            out[keys[0].split("_")[0] + "_" + result_key] = result
    return out
