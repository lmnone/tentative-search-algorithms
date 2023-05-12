import os.path
import pickle
import sys

import cv2
import numpy as np

from cyclic_list import CyclicList
from searchplan import SearchPlan
from binary_search_execution import LcpSearch, TargetDirSearchContext
from binary_search_execution import ExtremePointsFinder
from direction_information import DirectionInformation

"""
Find the longest chord, parallel to given direction
:param cyclic_list:
    A cyclic list of [x, y] pairs describing a closed, convex polygon. \ref cyclic_list
"""


def compute_longest_chord_parallel_to_given_direction(cyclic_list: CyclicList, target_dir: DirectionInformation,
                                                      refine=False) -> tuple:
    """
    Find the longest chord, parallel to given direction using prune-and-search algorithm
    :param target_dir: query direction
    :param refine:
    :param cyclic_list: A cyclic list of [x, y] pairs describing a closed, convex polygon.

    :return: tuple (AxisAlignedRectangle object, iterations count)
    """

    e = ExtremePointsFinder(cyclic_list)
    # print(e.extreme_points)

    lpd_configs = e.get_two_function_configs(type__=LcpSearch, context_type=TargetDirSearchContext,
                                             target_dir=target_dir, cleanup_nan_slopes=False)

    total_iterations = 0
    diagonal_plan = SearchPlan(cyclic_list, 'FULL', execution=lpd_configs['FULL'], refine=refine)
    diagonal_plan.apply_plan()
    total_iterations += diagonal_plan.iterations

    plan_config_dict = {}
    descr = diagonal_plan.descr()
    plan_config_dict[descr] = diagonal_plan

    return total_iterations, plan_config_dict


"""
Find the largest, inscribed, axis-aligned rectangle using tentative search algorithm
:param cyclic_list:
    A cyclic list of [x, y] pairs describing a closed, convex polygon. \ref cyclic_list
"""


def compute_largest_inscribed_isothetic_rectangle(cyclic_list: CyclicList, refine: bool = False,
                                                  debug: bool = False) -> tuple:
    """
    Find the largest, inscribed, axis-aligned rectangle using tentative search algorithm

    :param debug:
    :param refine:
    :param cyclic_list: A cyclic list of [x, y] pairs describing a closed, convex polygon.

    :return: tuple (AxisAlignedRectangle object, iterations count, canvas)
    Where canvas is 3-channel image used for debug purposes
    """
    e = ExtremePointsFinder(cyclic_list)
    # print(e.extreme_points)
    domain_configs = e.get_two_function_configs(debug=debug)
    print(domain_configs)

    plan_config_dict = {}
    total_iterations = 0
    for kind in filter(lambda s: 'FULL' not in s, domain_configs.keys()):
        plan = SearchPlan(cyclic_list, kind, execution=domain_configs[kind], refine=refine)
        plan.apply_plan()
        total_iterations += plan.iterations
        plan_config_dict[plan.descr()] = plan
        if debug:
            print('refine strength: %s, total iterations: %d' % (plan.refine_strength(), total_iterations))

    # other is outside bounds
    optimal_plans = list(
        filter(lambda x: x.optimal_solution().is_rectangle_in_polygon(cyclic_list, e.representative_point()),
               plan_config_dict.values()))

    if len(optimal_plans) < 1:
        return None, total_iterations, plan_config_dict, None

    optimal_plan = list(optimal_plans)[0]
    return optimal_plan.optimal_solution(), total_iterations, plan_config_dict, optimal_plan.descr()


if __name__ == '__main__':
    path = sys.argv[1]
    with open(path, 'rb') as handle:
        contours_orig = pickle.load(handle)

    print(type(contours_orig))
    c = CyclicList(contours_orig)

    output_path = '/Users/konstantinsobolev/Desktop/out'

    rc = compute_largest_inscribed_isothetic_rectangle(c, output_path, os.path.basename(path).replace('.pickle', ''),
                                                       refine=True)
    print(rc)
