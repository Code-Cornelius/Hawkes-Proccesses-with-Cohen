def multi_list_generator(size):
    """create the exact good list of size M*M*3


    Args:
        size:

    Returns: the function returns [   [0], [[0]], [[0]]   ] for size = 1.

    """
    if size < 1:
        raise Exception("Size not big enough.")
    ans = []
    interior_list = [0] * size
    # MU ALPHA BETA
    for j in range(3):
        medium_list = []  # list parameters
        for i in range(size):
            medium_list.append(interior_list.copy())  # not recreated every i so need copy.
            if j == 0 and i == 0:  # case for mu, reduces space used.
                medium_list = medium_list[0]
                break
        ans.append(medium_list)  # no copy because recreated every j
    return ans
