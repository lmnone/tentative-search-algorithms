from cyclic_list import CyclicList


def get_indexes(cyclic_list: CyclicList, start_: int, end_: int):
    indexes = []
    # inclusive [start,end]
    if end_ > start_:
        assert end_ > start_

        for i in range(start_, end_, 1):
            indexes.append(i)
    else:
        assert start_ > end_
        for i in range(start_, len(cyclic_list) + end_, 1):
            indexes.append(i % len(cyclic_list))

    return indexes


def get_total_polygon(cyclic_list: CyclicList, indexes: list):
    return list(map(lambda x: cyclic_list[x], indexes))


def get_slopes(cyclic_list: CyclicList, indexes: list):
    return list(map(lambda x: cyclic_list.direction_from_point(x).slope(), indexes))


def get_x(curve: list):
    # print(len(curve))
    return list(map(lambda x: x[0], curve))


def get_y(curve: list):
    # print(len(curve))
    return list(map(lambda x: x[1], curve))
