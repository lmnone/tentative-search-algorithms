import math

import numpy as np
from direction_information import DirectionInformation, point_of_segment_intersection
from cyclic_list import CyclicList


def refine_points_by_average_line_slope(cyclic_list: CyclicList, index_1: int, index_2, slope_target: float):
    # poly_1 = cyclic_list.direction_from_2_points(index_1, index_1 + 1)
    # poly_2 = cyclic_list.direction_from_2_points(index_2, index_2 + 1)

    assert np.isfinite(slope_target)

    cross_point = point_of_segment_intersection(cyclic_list[index_1], cyclic_list[index_2],
                                                cyclic_list[index_2 + 1], cyclic_list[index_1 + 1])

    dir_c1 = DirectionInformation(DirectionInformation.custom, cross_point, cyclic_list[index_1])
    dir_c2 = DirectionInformation(DirectionInformation.custom, cross_point, cyclic_list[index_1 + 1])
    # dir_c3 = DirectionInformation(DirectionInformation.custom, cross_point, cyclic_list[index_2])
    # dir_c4 = DirectionInformation(DirectionInformation.custom, cross_point, cyclic_list[index_2 + 1])

    slope_target = np.clip(slope_target, min(dir_c1.slope(), dir_c2.slope()), max(dir_c1.slope(), dir_c2.slope()))

    assert np.isfinite(slope_target)
    # y/x = slope
    view_x = max(cyclic_list[index_1][0], cyclic_list[index_2][0])
    line_point_1 = (np.array(cross_point) + np.array([view_x, slope_target * view_x])).tolist()
    line_point_2 = (np.array(cross_point) + np.array([-view_x, -slope_target * view_x])).tolist()
    # _dir = DirectionInformation(DirectionInformation.custom, line_point_1, line_point_2)
    cross_point_1 = point_of_segment_intersection(cyclic_list[index_1], cyclic_list[index_1 + 1], line_point_1,
                                                  line_point_2)
    cross_point_2 = point_of_segment_intersection(cyclic_list[index_2], cyclic_list[index_2 + 1], line_point_1,
                                                  line_point_2)

    diagonal_dir = DirectionInformation(DirectionInformation.custom, cross_point_1, cross_point_2)

    return cross_point_1, cross_point_2, diagonal_dir
