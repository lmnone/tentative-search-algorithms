import os.path
import pickle
import sys

import cv2
import numpy as np
from prettytable import PrettyTable

from binary_search_execution import ExtremePointsFinder
from cyclic_list import CyclicList
from pre_process import get_indexes
from searchplan import SearchPlan
from triangles import brute_force_get_maximal_3_corners_rc, draw_optimal_solution, minizinc_get_maximal_3_corners_rc


def draw_diagonal_dump(polygon__: list, canvas: np.array, thickness=1, color=(0, 0, 255)):
    cv2.polylines(canvas, [np.round(np.array(polygon__)).astype('int')], False, color=color)
    return


def draw_tangent(pt: list, slope, canvas__, thickness=1, color=(0, 0, 255), line_len=25):
    dx = line_len / np.sqrt(1 + slope ** 2)
    dy = slope * dx
    points = [pt, [pt[0] + dx, pt[1] + dy]]
    diagonal_points = np.round(np.array(points)).astype('int')
    cv2.arrowedLine(canvas__, diagonal_points[0], diagonal_points[1],
                    color, thickness=thickness)
    return


def draw_cross_point(p, canvas__: np.array, color=(0, 0, 255)):
    cv2.circle(canvas__, np.int0(p), 10, color, 2)


def get_x(curve: list):
    print(len(curve))
    return list(map(lambda x: x[0], curve))


def get_y(curve: list):
    print(len(curve))
    return list(map(lambda x: x[1], curve))


if __name__ == '__main__':
    path = sys.argv[1]
    dump_folder = sys.argv[2]

    with open(path, 'rb') as handle:
        contours_orig = pickle.load(handle)

    print(type(contours_orig))
    cyclic_list = CyclicList(contours_orig)
    print(cyclic_list)
    e = ExtremePointsFinder(cyclic_list)
    # print(e.extreme_points)
    liar_configs = e.get_three_function_configs(cleanup_nan_slopes=False)
    print(liar_configs.keys())
    t = PrettyTable(['filename', 'edges', 'area(minizinc)', 'area', 'iterations', 'iterations-tentative', 'config'])

    for config_descr in liar_configs.keys():
        # print(config_descr)
        plan = SearchPlan(cyclic_list, config_descr, execution=liar_configs[config_descr], refine=True)
        solution_object, solution_info = plan.apply_plan()
        # print((config_descr, plan.iterations, liar_configs[config_descr].context.function_space_counter))
        canvas = np.zeros([512, 512, 3], dtype=np.uint8)

        # plan.draw_found_domain_points(canvas)
        # plan.draw_optimal_solution(canvas)
        plan.draw_diagonal_dump(canvas)
        plan.draw_refine_boxes(canvas)
        plan.draw_cross_points(canvas)
        if solution_object is not None:
            # plan.draw_cross_points(canvas)
            plan.draw_optimal_solution(canvas)

        intervals = liar_configs[config_descr]
        a_indexes = get_indexes(cyclic_list, intervals.a_domain[0], intervals.a_domain[1])
        b_indexes = get_indexes(cyclic_list, intervals.b_domain[0], intervals.b_domain[1])
        c_indexes = get_indexes(cyclic_list, intervals.c_domain[0], intervals.c_domain[1])
        # a_indexes.reverse()
        # b_indexes.reverse()
        # c_indexes.reverse()

        # rc_near_optimal = brute_force_get_maximal_3_corners_rc(cyclic_list, e.representative_point(), a_indexes, b_indexes, c_indexes)

        rc_near_optimal = minizinc_get_maximal_3_corners_rc(config_descr, cyclic_list, e.representative_point(),
                                                            a_indexes,
                                                            b_indexes, c_indexes, canvas, draw_curvers=True)
        # plan.draw_polylines(canvas, use_markers=True)

        t.add_row(
            [os.path.basename(path), len(cyclic_list),
             np.round(rc_near_optimal.area(), 2) if rc_near_optimal is not None else 0,
             np.round(solution_object.area() if solution_object is not None else 0, 2), plan.iterations,
             liar_configs[config_descr].context.function_space_counter,
             plan.descr()])
        #
        # draw_optimal_solution(canvas, rc_near_optimal)

        filepath = os.path.join(dump_folder, plan.descr()) + '.jpg'
        cv2.imwrite(filepath, canvas)
    print(t)
