import os
import sys
import itertools
import numpy as np
from minizinc import Instance, Model, Solver
from pymzn import dzn
import enum
import pickle
import cv2
from binary_search_execution import ExtremePointsFinder
from cyclic_list import CyclicList
from pre_process import get_indexes, get_total_polygon, get_slopes, get_x, get_y
from axisalignedrectangle import axis_aligned_rectangle_from_2_point, AxisAlignedRectangle
from sortedcontainers import SortedList


def draw_curve(polygon__: list, canvas: np.array, thickness=1, color=(0, 0, 255)):
    cv2.polylines(canvas, [np.round(np.array(polygon__)).astype('int')], False, thickness=thickness, color=color)
    return


def draw_optimal_solution(canvas: np.array, optimal_rc, color=(250, 128, 128), thickness=1):
    assert type(optimal_rc) == AxisAlignedRectangle
    cv2.rectangle(canvas, np.int0(optimal_rc.bl), np.int0(optimal_rc.tr), color, thickness)
    return


def draw_tangent(pt: list, slope, canvas__, thickness=1, color=(0, 0, 255), line_len=25):
    dx = line_len / np.sqrt(1 + slope ** 2)
    dy = slope * dx
    points = [pt, [pt[0] + dx, pt[1] + dy]]
    print(pt)
    diagonal_points = np.round(np.array(points)).astype('int')
    cv2.arrowedLine(canvas__, diagonal_points[0], diagonal_points[1],
                    color, thickness=thickness)
    return


def draw_cross_point(p, canvas__: np.array, color=(0, 0, 255)):
    cv2.circle(canvas__, np.int0(p), 10, color, 2)


class TriangleSolver:
    config_2_sign = \
        {
            'SCAN_LEADING_BR': {'sign_target_ac_dx': -1,
                                'sign_target_ac_dy': 1,
                                'sign_slopes': 1,
                                'sign_m': 1
                                },
            'SCAN_LEADING_TR': {'sign_target_ac_dx': -1,
                                'sign_target_ac_dy': -1,
                                'sign_slopes': 1,
                                'sign_m': -1
                                },
            'SCAN_LEADING_BL': {'sign_target_ac_dx': 1,
                                'sign_target_ac_dy': 1,
                                'sign_slopes': 1,
                                'sign_m': -1
                                },
            'SCAN_LEADING_TL': {'sign_target_ac_dx': 1,
                                'sign_target_ac_dy': -1,
                                'sign_slopes': -1,
                                'sign_m': -1
                                }
        }

    def __init__(self, model_path: str = './minizinc'):
        print('loading %s' % model_path)
        self.exec_path = os.path.join(model_path, "triangle-in-convex.mzn")
        self.data_path = os.path.join(model_path, "triangle-data_custom.dzn")
        self.triangle_solver = Model(self.exec_path)
        self.optimizer = Solver.lookup("scip")

    def config_names(self) -> list:
        return list(self.config_2_sign.keys())

    def calc(self, a_x: list, a_y: list, b_x: list, b_y: list, c_x: list, c_y: list, slopes_a: list, slopes_b: list,
             slopes_c: list, config_descr: str) -> list:
        instance = Instance(self.optimizer, self.triangle_solver)
        # instance.add_file(self.data_path)
        assert len(a_x) == len(a_y)
        assert len(b_x) == len(b_y)
        assert len(c_x) == len(c_y)

        assert len(a_x) == len(slopes_a)
        assert len(b_x) == len(slopes_b)
        assert len(c_x) == len(slopes_c)

        instance['a_x'] = a_x
        instance['a_y'] = a_y
        instance['b_x'] = b_x
        instance['b_y'] = b_y
        instance['c_x'] = c_x
        instance['c_y'] = c_y
        #
        instance['a_slopes'] = slopes_a
        instance['b_slopes'] = slopes_b
        instance['c_slopes'] = slopes_c

        instance['a_len'] = len(a_x)
        instance['b_len'] = len(b_x)
        instance['c_len'] = len(c_x)

        # export vars to dzn
        data = {

            "a_x": a_x,
            "a_y": a_y,

            "b_x": b_x,
            "b_y": b_y,

            "c_x": c_x,
            "c_y": c_y,

            "a_slopes": slopes_a,
            "b_slopes": slopes_b,
            "c_slopes": slopes_c,

            "a_len": len(a_x),
            "b_len": len(b_x),
            "c_len": len(c_x),
        }

        # merge config
        for key, value in self.config_2_sign[config_descr].items():
            instance[key] = value
            data[key] = value

        with open(self.data_path, "w") as f:
            f.write("\n".join(dzn.dict2dzn(data)))

        result = instance.solve(intermediate_solutions=False)
        print('found %d solutions' % len(result))
        return result


def brute_force_get_maximal_3_corners_rc(cyclic_list_: CyclicList, representative_point: list, a_indexes_: list,
                                         b_indexes_: list,
                                         c_indexes_: list):
    scores = SortedList(key=lambda x: -x[0])
    for a_i in a_indexes_:
        # find near y in b
        matched_b_i = sorted(b_indexes_, key=lambda b_ind: abs(cyclic_list_[b_ind][1] - cyclic_list_[a_i][1]))[0]
        # find near x in c
        matched_c_i = sorted(c_indexes_, key=lambda c_ind: abs(cyclic_list_[c_ind][0] - cyclic_list_[matched_b_i][0]))[
            0]
        rc_ = axis_aligned_rectangle_from_2_point(cyclic_list_[a_i], cyclic_list_[matched_c_i])

        if not cyclic_list_.is_point_in_convex_polygon([cyclic_list_[matched_b_i][0], cyclic_list_[a_i][1]],
                                                       representative_point):
            continue
        if not cyclic_list_.is_point_in_convex_polygon([cyclic_list_[matched_c_i][0], cyclic_list_[a_i][1]],
                                                       representative_point):
            continue

        scores.add((rc_.area(), rc_))

    return scores[0][1]


def minizinc_get_maximal_3_corners_rc_preprocess(cyclic_list__: CyclicList, representative_point: list,
                                                 a_indexes__: list,
                                                 b_indexes__: list,
                                                 c_indexes__: list, canvas,
                                                 calc_brute_force: bool = True,
                                                 draw_curves: bool = True):
    """

    :param draw_curves:
    :param calc_brute_force:
    :param cyclic_list__:
    :param representative_point:
    :param a_indexes__:
    :param b_indexes__:
    :param c_indexes__:
    :param canvas:
    :return:
    """

    if calc_brute_force:
        rc_near_optimal = brute_force_get_maximal_3_corners_rc(cyclic_list__, representative_point, a_indexes__,
                                                               b_indexes__,
                                                               c_indexes__)
        draw_optimal_solution(canvas, rc_near_optimal, color=(0, 0, 125), thickness=4)
        print('best(brute_force) area: %f' % rc_near_optimal.area())

    polygon_a = get_total_polygon(cyclic_list__, a_indexes__)
    polygon_b = get_total_polygon(cyclic_list__, b_indexes__)
    polygon_c = get_total_polygon(cyclic_list__, c_indexes__)

    slopes_a = get_slopes(cyclic_list__, a_indexes__)
    slopes_b = get_slopes(cyclic_list__, b_indexes__)
    slopes_c = get_slopes(cyclic_list__, c_indexes__)

    a_x = get_x(polygon_a)
    a_y = get_y(polygon_a)
    b_x = get_x(polygon_b)
    b_y = get_y(polygon_b)
    c_x = get_x(polygon_c)
    c_y = get_y(polygon_c)

    if draw_curves:
        draw_curve(polygon_a, canvas, color=(0, 0, 255), thickness=1)
        draw_curve(polygon_b, canvas, color=(0, 0, 255), thickness=2)
        draw_curve(polygon_c, canvas, color=(0, 0, 255), thickness=3)

    return a_x, a_y, b_x, b_y, c_x, c_y, slopes_a, slopes_b, slopes_c


def minizinc_get_maximal_3_corners_rc(config_descr_: str, cyclic_list_: CyclicList, representative_point: list, a_indexes_: list,
                                      b_indexes_: list,
                                      c_indexes_: list, canvas,
                                      draw_minizinc_tangents: bool = True,
                                      draw_minizinc_solution: bool = True,
                                      draw_curvers: bool = False) -> AxisAlignedRectangle:
    a_x, a_y, b_x, b_y, c_x, c_y, slopes_a, slopes_b, slopes_c = \
        minizinc_get_maximal_3_corners_rc_preprocess(cyclic_list_, representative_point, a_indexes_, b_indexes_,
                                                     c_indexes_, canvas, draw_curves=draw_curvers, calc_brute_force=False)

    triangle_solver = TriangleSolver()
    solutions = triangle_solver.calc(a_x, a_y, b_x, b_y, c_x, c_y, slopes_a, slopes_b, slopes_c, config_descr_)

    solution = solutions.solution
    if solution is None:
        return None

    a = (solution.a_x_refine, solution.a_y_refine)
    b = (solution.b_x_refine, solution.b_y_refine)
    c = (solution.c_x_refine, solution.c_y_refine)

    # minizinc index is 1-based
    a_i = solution.a - 1
    b_i = solution.b - 1
    c_i = solution.c - 1

    m_a = slopes_a[a_i]
    m_b = slopes_b[b_i]
    m_c = slopes_c[c_i]

    if draw_minizinc_tangents:
        draw_tangent(a, m_a, canvas, color=(0, 210, 195))

        draw_tangent(b, m_b, canvas, color=(0, 210, 195))

        draw_tangent(c, m_c, canvas, color=(0, 210, 195))

    rc = axis_aligned_rectangle_from_2_point(a, c)

    if draw_minizinc_solution:
        draw_optimal_solution(canvas, rc)

    return rc


if __name__ == '__main__':
    path = sys.argv[2]
    with open(path, 'rb') as handle:
        contours_orig = pickle.load(handle)

    dump_folder = sys.argv[3]

    print(type(contours_orig))
    cyclic_list = CyclicList(contours_orig)

    e = ExtremePointsFinder(cyclic_list)
    # print(e.extreme_points)
    liar_configs = e.get_three_function_configs(cleanup_nan_slopes=False)
    print(liar_configs)

    print(cyclic_list)

    model_path = sys.argv[1]

    triangle_solver = TriangleSolver()

    for config_descr in triangle_solver.config_names():
        print(config_descr)
        canvas = np.zeros([512, 512, 3], dtype=np.uint8)

        intervals = liar_configs[config_descr]
        a_indexes = get_indexes(cyclic_list, intervals.a_domain[0], intervals.a_domain[1])
        b_indexes = get_indexes(cyclic_list, intervals.b_domain[0], intervals.b_domain[1])
        c_indexes = get_indexes(cyclic_list, intervals.c_domain[0], intervals.c_domain[1])

        a_x, a_y, b_x, b_y, c_x, c_y, slopes_a, slopes_b, slopes_c = \
            minizinc_get_maximal_3_corners_rc_preprocess(cyclic_list, e.representative_point(), a_indexes, b_indexes,
                                                         c_indexes, canvas)

        solutions = triangle_solver.calc(a_x, a_y, b_x, b_y, c_x, c_y, slopes_a, slopes_b, slopes_c, config_descr)
        for solution in [solutions.solution]:
            m_ac = solution.m_ac
            a = (solution.a_x_refine, solution.a_y_refine)
            b = (solution.b_x_refine, solution.b_y_refine)
            c = (solution.c_x_refine, solution.c_y_refine)

            # minizinc index is 1-based
            a_i = solution.a - 1
            b_i = solution.b - 1
            c_i = solution.c - 1
            print('=============')
            print('m_a:{},m_c:{} => m_ac:{}'.format(slopes_a[a_i], slopes_c[c_i], m_ac))

            m_a = slopes_a[a_i]
            m_b = slopes_b[b_i]
            m_c = slopes_c[c_i]

            print('diff_y:{}, diff_x:{}, m_ac(calculated):{}'.format(a[1] - b[1], b[0] - c[0],
                                                                     float(c[1] - a[1]) / float(c[0] - a[0])))

            draw_tangent(a, m_a, canvas, color=(0, 210, 195))

            draw_tangent(b, m_b, canvas, color=(0, 210, 195))

            draw_tangent(c, m_c, canvas, color=(0, 210, 195))

            rc = axis_aligned_rectangle_from_2_point(a, c)
            print(rc.area())
            draw_optimal_solution(canvas, rc)

        filepath = os.path.join(dump_folder, config_descr + '.minizinc.') + '.jpg'
        cv2.imwrite(filepath, canvas)
