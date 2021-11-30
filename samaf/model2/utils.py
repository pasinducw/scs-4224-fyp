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
