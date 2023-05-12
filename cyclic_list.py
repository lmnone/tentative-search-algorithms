import typing

import numpy as np
from typing import Type
from direction_information import DirectionInformation


class CyclicList(list):
    def __getitem__(self, index):
        if index < 0:
            index += len(self)
        return super(CyclicList, self).__getitem__(index % len(self))

    def __repr__(self):
        return ','.join(map(lambda x: '[{},{}] '.format(x[0], x[1]), self))

    def find_by_coordinate(self, Z, direction_information: DirectionInformation) -> list:
        indexes = []
        assert len(Z) == 2

        # \todo custom part not tested
        if direction_information.pc == DirectionInformation.custom:
            for i in range(0, len(self)):
                if self[i][0] * direction_information.a1 + self[i][1] * direction_information.a2 \
                        == direction_information.b:
                    indexes.append((True, i))
                    continue
                elif np.sign(self[i][0] * direction_information.a1 + self[i][1] * direction_information.a2
                             - direction_information.b) != \
                        np.sign(self[i + 1][0] * direction_information.a1 + self[i + 1][1] * direction_information.a2
                                - direction_information.b):

                    indexes.append((False, i, i + 1))
                    continue
        else:
            dimension_order = direction_information.query_dimension_order()
            z = Z[dimension_order]
            for i in range(0, len(self)):
                if self[i][dimension_order] == z:
                    indexes.append((True, i))
                    continue
                if self[i][dimension_order] < z < self[i + 1][dimension_order]:
                    indexes.append((False, i, i + 1))
                    continue
                if self[i][dimension_order] > z > self[i + 1][dimension_order]:
                    indexes.append((False, i + 1, i))
                    continue

        return indexes

    def is_point_in_convex_polygon(self, point: list, representative_point: list) -> bool:

        cyclic_list = self

        def metric_1(index):
            dir_ = self.direction_from_point(index)
            # found on-line
            if dir_.is_on_line(point):
                return None

            p1 = cyclic_list[index]
            p2 = cyclic_list[index + 1]
            center_of_interval = ((np.array(p1) + np.array(p2)) / 2)
            dir_2_target = DirectionInformation(DirectionInformation.custom, list(center_of_interval), point)

            assert dir_.length() > 0
            assert dir_2_target.length() > 0
            return dir_2_target.dot_normed(dir_)

        polygon = self
        # Find the edge above or below the point using binary search
        lo = 0
        hi = len(polygon)
        while lo < hi:
            mid = (lo + hi) // 2

            score = metric_1(mid)
            if score is None:
                # found on-line
                return True

            if score <= 0:
                hi = mid
            else:
                lo = mid + 1
        # Check if point is to the left or right of the edge
        found = lo
        dir_chunk = self.direction_from_point(found)

        if dir_chunk.is_on_line(point):
            return True

        return dir_chunk.is_same_halfplane(representative_point, point)

    def binary_search_max_y_vertex(self):
        polygon = self
        n = len(polygon)
        lo, hi = 0, n - 1

        while lo <= hi:
            mid = (lo + hi) // 2
            if polygon[mid][1] >= polygon[(mid + 1)][1]:
                hi = mid - 1
            else:
                lo = mid + 1

        return lo

    def binary_search_min_y_vertex(self):
        polygon = self
        n = len(polygon)
        lo, hi = 0, n - 1

        while lo <= hi:
            mid = (lo + hi) // 2
            if polygon[mid][1] <= polygon[(mid + 1)][1]:
                hi = mid - 1
            else:
                lo = mid + 1

        return lo

    def binary_search_max_x_vertex(self):
        polygon = self
        n = len(polygon)
        lo, hi = 0, n - 1

        while lo <= hi:
            mid = (lo + hi) // 2
            if polygon[mid][0] >= polygon[(mid + 1)][0]:
                hi = mid - 1
            else:
                lo = mid + 1

        return lo

    def binary_search_min_x_vertex(self):
        polygon = self
        n = len(polygon)
        lo, hi = 0, n - 1

        while lo <= hi:
            mid = (lo + hi) // 2
            if polygon[mid][0] <= polygon[(mid + 1)][0]:
                hi = mid - 1
            else:
                lo = mid + 1

        return lo

    def direction_from_point(self, index: int):
        cyclic_list = self
        _dir = DirectionInformation(DirectionInformation.custom, cyclic_list[index],
                                    cyclic_list[index + 1])
        return _dir

    def direction_from_2_points(self, index_1: int, index_2):
        cyclic_list = self
        _dir = DirectionInformation(DirectionInformation.custom, cyclic_list[index_1],
                                    cyclic_list[index_2])
        return _dir

    def get_edge_slope(self, i: int):
        cyclic_list = self
        _dir = DirectionInformation(DirectionInformation.custom, cyclic_list[i], cyclic_list[i + 1])
        return _dir.slope()


class TransformedCyclicList(CyclicList):

    def __init__(self, c: CyclicList, trans: np.array):
        super().__init__(c)
        self.trans = trans

    def __getitem__(self, index):
        xy = super().__getitem__(index)
        return np.dot(self.trans, xy).tolist()
