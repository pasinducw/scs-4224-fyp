import torch


def combine_dimensions(tensor: torch.Tensor, a: int, b: int):
    others = []
    for index in range(tensor.dim()):
        if index != a and index != b:
            others.append(index)

    permutation_indices = [*others, a, b]
    tensor = tensor.permute(permutation_indices)

    sizes = []
    for index in range(tensor.dim()-2):
        sizes.append(tensor.shape[index])

    tensor = tensor.view(*sizes, -1)

    initials = []
    for index in range(tensor.dim()-1):
        initials.append(index)

    tensor = tensor.permute(tensor.dim()-1, *initials)
    return tensor


def lower_bound(sorted_elements: list, key: None, value: None):
    n = len(sorted_elements)
    k = -1
    b = n
    while (b >= 1):
        while (k+b < n and key(sorted_elements[k+b]) < value):
            k += b
        b = b//2
    return k+1


def upper_bound(sorted_elements: list, key: None, value: None):
    n = len(sorted_elements)
    k = -1
    b = n
    while (b >= 1):
        while (k+b < n and key(sorted_elements[k+b]) <= value):
            k += b
        b = b//2
    return k+1
