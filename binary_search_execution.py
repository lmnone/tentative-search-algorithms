import os.path
import pickle
import sys
from typing import Dict, Tuple, List
import cv2
import numpy as np
import itertools

from axisalignedrectangle import axis_aligned_rectangle_from_2_point
from cyclic_list import CyclicList, TransformedCyclicList
from pre_process import get_indexes
from direction_information import DirectionInformation, point_of_segment_intersection
from refine_points import refine_points_by_average_line_slope


class GenericContext:
    """
    Custom storage per-execution instance
    """

    def __init__(self, diagonal_dump=None, dump_type: str = 'generic', debug: bool = False):
        if diagonal_dump is None:
            self.diagonal_dump = []
        else:
            assert type(diagonal_dump) == list
            self.diagonal_dump = diagonal_dump

        self.dump_type = dump_type
        self.debug = debug


class TargetDirSearchContext(GenericContext):
    def __init__(self, target_dir: DirectionInformation, diagonal_dump=None):
        super().__init__(diagonal_dump, dump_type='diagonal_search')
        self.diagonal_dump.append(target_dir)
        self.target_dir = target_dir


class GenericExecution:
    """
    Abstract class for fixed-point & tentative search for monotonic function composition
    """

    def __init__(self, n: int, context: GenericContext):
        """

        :param n:
            Vertexes number in convex polygon
        """

        assert type(n) == int
        self.n = n
        assert issubclass(type(context), GenericContext)
        self.context = context
        self.debug = context.debug

    def get_interval_len_inclusive_exclusive(self, __i2: int, __i1: int) -> int:
        """
            Interval len getter
        :param __i2: right index of cyclic list
        :param __i1: left index of cyclic list
        :return: interval length, order of index parameters at input does not matter
        """
        __delta = __i2 - __i1
        if __delta < 0:
            __delta += self.n
        return __delta

    def is_in_domain(self, index: int, domain: tuple):
        """
            Check whether specific index is in domain (for ex. a_domain, b_domain, ...)
        :param index:
        :param domain: tuple (i1, i2) of cyclic list
        :return: True if index is in domain, else False
        """
        if domain[0] <= domain[1]:
            return domain[0] <= index <= domain[1]
        else:
            assert index >= self.n
            return domain[1] <= index <= domain[0]

    def search_core(self,
                    cyclic_list: CyclicList,
                    stack: int = 0) -> tuple:
        """
            Main search algorithm prototype
        :param cyclic_list: Input polygon (convex)
        :param stack: Variable to control iteration number. Should be 0 on start.
        """
        pass

    def __lshift__(self, domain_descr: str):
        """
            Shift interval down on specific domain (equivalent to binary search iteration)
        :param domain_descr: name of domain e.t.c. 'f','g','h'
        """
        pass

    def __rshift__(self, domain_descr: str):
        """
            Shift interval up on specific domain (equivalent to binary search iteration)
        :param domain_descr: name of domain e.t.c. 'f','g','h'
        """
        pass

    # get indexes from all partitions
    def retrieve_indexes(self, cyclic_list: CyclicList):
        """
            Retrieve indexes from cyclic list according intervals.
            Should be K-list for K-composition search
        :param cyclic_list: convex polygon
        """
        pass

    def get_solution_and_score_from_found_domain(self, solution_points: tuple) -> tuple:
        """
            Then applying some refine algorithm this function incorporate score calculate (should be overloaded)
            :param solution_points: prune & search solution, could be 2 points
            or 3 points for 2 functions and 3 functions fixed point search respectively
        """
        pass

    def refine_solution(self, cyclic_list: CyclicList, found_domain: tuple, shifts: tuple = None) -> tuple:
        pass


class PruneSearchTwoFunctions(GenericExecution):
    # f,g => lo,up,lo,up
    execution_inc_dec = {
        (1, 1): [0, 0, 0, 1],
        (1, 0): [1, 0, 0, 0],
        (0, 1): [0, 1, 0, 0],
        (0, 0): [0, 0, 1, 0]
    }

    execution_dec_inc = {
        (1, 1): [0, 1, 0, 0],
        (1, 0): [0, 0, 0, 1],
        (0, 1): [0, 0, 1, 0],
        (0, 0): [1, 0, 0, 0]
    }

    def __init__(self, f_domain: tuple, g_domain: tuple, n: int, context: GenericContext):

        super().__init__(n, context)

        assert type(f_domain) == tuple and type(g_domain) == tuple
        self.f_domain = f_domain
        self.g_domain = g_domain

        self.f_len, self.g_len = self.get_interval_len_inclusive_exclusive(self.f_domain[1],
                                                                           self.f_domain[
                                                                               0]), self.get_interval_len_inclusive_exclusive(
            self.g_domain[1],
            self.g_domain[0])

        self.f_mid_index = (self.f_domain[0] + self.f_len // 2) % self.n
        self.g_mid_index = (self.g_domain[0] + self.g_len // 2) % self.n

    @staticmethod
    def decode_domain_update(update: list):
        nz_index = update.index(1)
        _function = 'g' if nz_index // 2 > 0 else 'f'
        __lo_up = 'up' if nz_index % 2 == 1 else 'lo'
        return _function, __lo_up

    def __repr__(self):
        return 'f,g:  {} -> {}'.format(self.f_domain, self.g_domain)

    def __str__(self) -> str:
        return 'f,g:  {} -> {}'.format(self.f_domain, self.g_domain)

    def __lshift__(self, domain_descr: str):
        # assert domain_descr == 'f' or domain_descr == 'g'
        if domain_descr == 'f':
            return type(self)((self.f_domain[0], self.f_mid_index), self.g_domain, self.n,
                              self.context)
        elif domain_descr == 'g':
            return type(self)(self.f_domain, (self.g_domain[0], self.g_mid_index), self.n,
                              self.context)
        else:
            assert False

    def __rshift__(self, domain_descr: str):
        # assert domain_descr == 'f' or domain_descr == 'g'
        if domain_descr == 'f':
            return type(self)((self.f_mid_index, self.f_domain[1]), self.g_domain, self.n,
                              self.context)
        elif domain_descr == 'g':
            return type(self)(self.f_domain, (self.g_mid_index, self.g_domain[1]), self.n,
                              self.context)
        else:
            assert False

    def retrieve_indexes(self, cyclic_list: CyclicList):
        return get_indexes(cyclic_list, self.f_domain[0], self.f_domain[1]), get_indexes(cyclic_list, self.g_domain[0],
                                                                                         self.g_domain[1])

    def search_core(self,
                    cyclic_list: CyclicList,
                    stack: int = 0) -> tuple:
        pass


class ThreeFunctionsSearchContext(GenericContext):
    MAX_TENTATIVE_ITERATIONS = 15
    MAX_ITERATIONS = 15

    def __init__(self, config_descr: str, diagonal_dump=None):
        super().__init__(diagonal_dump)

        self.config_descr = config_descr

        self.sign_config = {'SCAN_LEADING_BL': {'metric_1:slope_sign': 1, 'm_ac_slope:diagonal_sign': 1,
                                                'm_a__m_c:compare': 1,
                                                'control:m_a_sign': -1,
                                                'control:m_b_sign': 1, 'control:m_c_sign': -1, 'score_g:x_sign': 1,
                                                'score_f:y_sign': -1,
                                                'refine:b_top_bottom': -1, 'refine:a_left_right': -1,
                                                'refine:a_dir': DirectionInformation(DirectionInformation.custom,
                                                                                     [0, 0], [0, -1]),
                                                'refine:b_dir': DirectionInformation(DirectionInformation.custom,
                                                                                     [0, 0], [0, 1]),
                                                'refine:c_dir': DirectionInformation(DirectionInformation.custom,
                                                                                     [0, 0], [0, 1]),
                                                },

                            'SCAN_LEADING_TL': {'metric_1:slope_sign': -1, 'm_ac_slope:diagonal_sign': -1,
                                                'm_a__m_c:compare': -1,
                                                'control:m_a_sign': -1,
                                                'control:m_b_sign': 1, 'control:m_c_sign': -1, 'score_g:x_sign': -1,
                                                'score_f:y_sign': -1,
                                                'refine:b_top_bottom': 1, 'refine:a_left_right': -1,
                                                'refine:a_dir': DirectionInformation(DirectionInformation.custom,
                                                                                     [0, 0], [0, -1]),
                                                'refine:b_dir': DirectionInformation(DirectionInformation.custom,
                                                                                     [0, 0], [0, 1]),
                                                'refine:c_dir': DirectionInformation(DirectionInformation.custom,
                                                                                     [0, 0], [0, 1]),
                                                },

                            'SCAN_LEADING_BR': {'metric_1:slope_sign': -1, 'm_ac_slope:diagonal_sign': -1,
                                                'm_a__m_c:compare': -1,  # ?
                                                'control:m_a_sign': -1,
                                                'control:m_b_sign': 1, 'control:m_c_sign': -1, 'score_g:x_sign': 1,
                                                'score_f:y_sign': 1,
                                                'refine:b_top_bottom': -1, 'refine:a_left_right': 1,
                                                'refine:a_dir': DirectionInformation(DirectionInformation.custom,
                                                                                     [0, 0], [0, 1]),
                                                'refine:b_dir': DirectionInformation(DirectionInformation.custom,
                                                                                     [0, 0], [0, -1]),
                                                'refine:c_dir': DirectionInformation(DirectionInformation.custom,
                                                                                     [0, 0], [0, -1]),
                                                },
                            'SCAN_LEADING_TR': {'metric_1:slope_sign': 1, 'm_ac_slope:diagonal_sign': 1,
                                                'm_a__m_c:compare': 1,
                                                'control:m_a_sign': -1,
                                                'control:m_b_sign': 1, 'control:m_c_sign': -1, 'score_g:x_sign': -1,
                                                'score_f:y_sign': 1,
                                                'refine:b_top_bottom': 1, 'refine:a_left_right': 1,
                                                'refine:a_dir': DirectionInformation(DirectionInformation.custom,
                                                                                     [0, 0], [0, 1]),
                                                'refine:b_dir': DirectionInformation(DirectionInformation.custom,
                                                                                     [0, 0], [0, -1]),
                                                'refine:c_dir': DirectionInformation(DirectionInformation.custom,
                                                                                     [0, 0], [0, -1]),
                                                }}

        self.function_space_counter = 0
        self.tentative_stack = []

        self.dump_type = 'ThreeFunctionsSearchContext'
        # debug info
        self.refine_boxes = []
        self.cross_point_with_borders = []

    def get_sign_config(self) -> dict:
        return self.sign_config[self.config_descr]


class TentativeSearchThreeFunctions(GenericExecution):
    def refine_solution(self, cyclic_list: CyclicList, found_domain: tuple, shifts: tuple = None) -> tuple:
        a_i, b_i, c_i = found_domain
        context: ThreeFunctionsSearchContext = self.context

        if shifts is None:
            a_shift, b_shift, c_shift = 1, 1, 1
        else:
            a_shift, b_shift, c_shift = shifts

        def is_chunk_in_domain(chunk: tuple, domain: tuple):
            return self.is_in_domain(chunk[0], domain) and self.is_in_domain(chunk[1], domain)

        if not (is_chunk_in_domain((a_i, a_i + a_shift), self.a_domain) and
                is_chunk_in_domain((b_i, b_i + b_shift), self.b_domain) and
                is_chunk_in_domain((c_i, c_i + c_shift), self.c_domain)):
            return None, None, None

        a_box = axis_aligned_rectangle_from_2_point(cyclic_list[a_i], cyclic_list[a_i + a_shift])
        b_box = axis_aligned_rectangle_from_2_point(cyclic_list[b_i], cyclic_list[b_i + b_shift])
        c_box = axis_aligned_rectangle_from_2_point(cyclic_list[c_i], cyclic_list[c_i + c_shift])

        # boxes without intersection => no solution
        b_box_updated = b_box.interval_intersection_xy(c_box.x_lo_hi, a_box.y_lo_hi)
        if b_box_updated is None:
            return None, None, None

        def metric_2(cyclic_list_, a, b):
            return cyclic_list_.direction_from_2_points(a, b)

        a_dir = metric_2(cyclic_list, a_i, a_i + a_shift)
        b_dir = metric_2(cyclic_list, b_i, b_i + b_shift)
        c_dir = metric_2(cyclic_list, c_i, c_i + c_shift)

        cross_points_with_borders = list(b_box_updated.cross_point_with_borders(b_dir))

        def get_a_c_cross_points(x, y):
            _BIG_VIEW = 500
            b_x_axis = DirectionInformation(DirectionInformation.custom, [x, y],
                                            [x + _BIG_VIEW, y])

            b_y_axis = DirectionInformation(DirectionInformation.custom, [x, y],
                                            [x, y - _BIG_VIEW])

            # self.context.diagonal_dump.append(b_x_axis)
            # self.context.diagonal_dump.append(b_y_axis)

            cross_point_a_ = point_of_segment_intersection(b_x_axis.pt1, b_x_axis.pt2, cyclic_list[a_i],
                                                           cyclic_list[a_i + a_shift])
            cross_point_c_ = point_of_segment_intersection(b_y_axis.pt1, b_y_axis.pt2, cyclic_list[c_i],
                                                           cyclic_list[c_i + c_shift])

            W_ = (cross_point_a_[0] - x) * self.sign_config['refine:a_left_right']
            H_ = (y - cross_point_c_[1]) * self.sign_config['refine:b_top_bottom']
            assert W_ > 0
            assert H_ > 0

            return cross_point_a_, cross_point_c_, W_, H_

        # b-chunk and updated b-box have no intersection ( Why?)
        if len(cross_points_with_borders) == 0:
            if self.debug:
                context.refine_boxes.append(b_box_updated)
                self.context.diagonal_dump.append(b_dir)
            return None, None, None

        # updated b-point found exactly, just return
        elif len(cross_points_with_borders) == 1:
            refined_b = cross_points_with_borders[0]
            cross_point_a, cross_point_c, _, __ = get_a_c_cross_points(refined_b[0], refined_b[1])
        # Line found
        else:
            assert len(cross_points_with_borders) == 2

            cross_points_with_borders = sorted(cross_points_with_borders,
                                               key=lambda x: self.sign_config['refine:b_top_bottom'] * x[1])

            b_box_updated__outer_point = list(cross_points_with_borders[-1])
            b_box_updated__inner_point = list(cross_points_with_borders[0])

            if self.debug:
                context.cross_point_with_borders.append(b_box_updated__outer_point)
                context.refine_boxes.append(b_box_updated)
                context.refine_boxes.append(a_box)
                context.refine_boxes.append(c_box)

            x_hi, y_hi = b_box_updated__outer_point
            x_lo, y_lo = b_box_updated__inner_point
            box_length = np.linalg.norm([x_hi - x_lo, y_hi - y_lo])

            cross_point_a, cross_point_c, W, H = get_a_c_cross_points(x_hi, y_hi)
            cross_point_a, cross_point_c, W1, H1 = get_a_c_cross_points(x_lo, y_lo)
            assert H >= H1
            assert W <= W1

            def normalize_angle(box_direction: DirectionInformation, axis: DirectionInformation, shift: int = 1):
                fi = np.arccos(box_direction.dot_normed(axis) * shift)
                assert 0 <= fi <= np.pi / 2
                return fi

            fi_1 = normalize_angle(b_dir, self.sign_config['refine:b_dir'], b_shift)
            fi_2 = normalize_angle(c_dir, self.sign_config['refine:c_dir'], c_shift)
            fi_3 = normalize_angle(a_dir, self.sign_config['refine:a_dir'], a_shift)

            p = np.sin(fi_1) + np.cos(fi_1) * np.tan(fi_2)
            q = np.cos(fi_1) + np.sin(fi_1) / np.tan(fi_3)

            t = (H * q - W * p) / (2 * p * q)
            t = min(max(t, 0), box_length) / box_length
            print('t: %f' % t)
            # t = 0
            refined_b = np.array([x_hi, y_hi]) * (1.0 - t) + t * np.array([x_lo, y_lo])

            cross_point_a, cross_point_c, _, __ = get_a_c_cross_points(refined_b[0], refined_b[1])

        return cross_point_a, [refined_b[0], refined_b[1]], cross_point_c

    # this table contains logical resolvable key configurations ( non -tentative)
    execution_3x_dec_stable = {
        (1, 1, 1): [0, 0, 0, 0, 0, 0],
        (1, 1, 0): [0, 0, 0, 1, 0, 0],
        (1, 0, 1): [0, 1, 0, 0, 0, 0],
        (1, 0, 0): [0, 0, 0, 0, 1, 0],
        (0, 1, 1): [0, 0, 0, 0, 0, 1],
        (0, 1, 0): [1, 0, 0, 0, 0, 0],
        (0, 0, 1): [0, 0, 1, 0, 0, 0],
        (0, 0, 0): [0, 0, 0, 0, 0, 0]
    }

    execution_inc_dec = {
        (1, 1): [0, 0, 0, 1],
        (1, 0): [1, 0, 0, 0],
        (0, 1): [0, 1, 0, 0],
        (0, 0): [0, 0, 1, 0]
    }

    execution_dec_inc = {
        (1, 1): [0, 1, 0, 0],
        (1, 0): [0, 0, 0, 1],
        (0, 1): [0, 0, 1, 0],
        (0, 0): [1, 0, 0, 0]
    }

    def __init__(self, a_domain: tuple, b_domain: tuple, c_domain: tuple, n: int, context: ThreeFunctionsSearchContext):
        super().__init__(n, context)

        assert type(context) == ThreeFunctionsSearchContext
        self.sign_config = context.get_sign_config()

        assert type(a_domain) == tuple and type(b_domain) == tuple and type(c_domain) == tuple
        self.a_domain = a_domain
        self.b_domain = b_domain
        self.c_domain = c_domain

        self.a_len, self.b_len, self.c_len = self.get_interval_len_inclusive_exclusive(self.a_domain[1],
                                                                                       self.a_domain[
                                                                                           0]), self.get_interval_len_inclusive_exclusive(
            self.b_domain[1],
            self.b_domain[0]), self.get_interval_len_inclusive_exclusive(
            self.c_domain[1],
            self.c_domain[0])

        self.a_mid_index = (self.a_domain[0] + self.a_len // 2) % self.n
        self.b_mid_index = (self.b_domain[0] + self.b_len // 2) % self.n
        self.c_mid_index = (self.c_domain[0] + self.c_len // 2) % self.n

    def retrieve_indexes(self, cyclic_list: CyclicList):
        """
            This function only for debug usage
        :param cyclic_list:
        :return:
        """
        return get_indexes(cyclic_list, self.a_domain[0], self.a_domain[1]), get_indexes(cyclic_list,
                                                                                         self.b_domain[0],
                                                                                         self.b_domain[1]), get_indexes(
            cyclic_list,
            self.c_domain[0],
            self.c_domain[1])

    @property
    def spaces(self):
        return ['f', 'g', 'h']

    @property
    def spaces_degraded(self):
        domain_lens = {'f': self.a_len, 'g': self.b_len, 'h': self.c_len}
        degraded = (np.array(list(domain_lens.values())) <= 2)
        return degraded

    @property
    def spaces_filtered(self):
        domain_lens = {'f': self.a_len, 'g': self.b_len, 'h': self.c_len}
        return list(filter(lambda x: domain_lens[x] >= 2, self.spaces))

    def decode_domain_update(self, key: tuple):
        update = self.execution_3x_dec_stable[key]

        # domain_lens = {'f': self.a_len, 'g': self.b_len, 'h': self.c_len}

        if self.spaces_degraded.sum() == 1:

            skip_index = list(self.spaces_degraded).index(True)
            skip = self.spaces[skip_index]
            spaces_2 = self.spaces.copy()
            spaces_2.remove(skip)
            spaces_2 = tuple(spaces_2)
            ##########
            if spaces_2 == ('f', 'h'):
                # | f | X | h |
                # | INC | DEC
                key_2 = list(key)
                del key_2[skip_index]
                key_2 = tuple(key_2)
                update = self.execution_inc_dec[key_2]
                nz_index = update.index(1)

                _function = 'f' if nz_index // 2 > 0 else 'h'
                _lo_up = 'up' if nz_index % 2 == 1 else 'lo'

            elif spaces_2 == ('f', 'g'):
                # | f | g | X
                # | g | X | f
                # | INC | DEC
                # assert False

                key_2 = list(key)
                del key_2[skip_index]
                key_2 = tuple(key_2)
                update = self.execution_dec_inc[key_2]

                nz_index = update.index(1)
                _function = 'f' if nz_index // 2 > 0 else 'g'
                _lo_up = 'up' if nz_index % 2 == 1 else 'lo'
            else:
                # | X | g | h

                # | INC | DEC
                # assert False

                key_2 = list(key)
                del key_2[skip_index]
                key_2 = tuple(key_2)
                update = self.execution_inc_dec[key_2]

                nz_index = update.index(1)
                _function = 'g' if nz_index // 2 > 0 else 'h'
                _lo_up = 'up' if nz_index % 2 == 0 else 'lo'

        elif self.spaces_degraded.sum() == 2:
            # spaces = ['f', 'g', 'h']
            _index = list(self.spaces_degraded).index(False)
            _function = self.spaces[_index]
            _lo_up = 'up' if key[_index] == 1 else 'lo'
        else:
            nz_index = update.index(1)
            _function = ['f', 'g', 'h'][nz_index // 2]
            _lo_up = 'up' if nz_index % 2 == 1 else 'lo'
        return _function, _lo_up

    def __repr__(self):
        return 'f,g,h:  {} -> {} -> {}'.format(self.a_domain, self.b_domain, self.c_domain)

    def __str__(self) -> str:
        return 'f,g,h:  {} -> {} -> {}'.format(self.a_domain, self.b_domain, self.c_domain)

    def __lshift__(self, domain_descr: str):
        # assert domain_descr == 'f' or domain_descr == 'g' domain_descr == 'h'
        if domain_descr == 'f':
            return type(self)((self.a_domain[0], self.a_mid_index), self.b_domain, self.c_domain, self.n,
                              self.context)
        elif domain_descr == 'g':
            return type(self)(self.a_domain, (self.b_domain[0], self.b_mid_index), self.c_domain, self.n,
                              self.context)
        elif domain_descr == 'h':
            return type(self)(self.a_domain, self.b_domain, (self.c_domain[0], self.c_mid_index), self.n,
                              self.context)
        else:
            assert False

    def __rshift__(self, domain_descr: str):
        # assert domain_descr == 'f' or domain_descr == 'g' or domain_descr == 'h'
        if domain_descr == 'f':
            return type(self)((self.a_mid_index, self.a_domain[1]), self.b_domain, self.c_domain, self.n,
                              self.context)
        elif domain_descr == 'g':
            return type(self)(self.a_domain, (self.b_mid_index, self.b_domain[1]), self.c_domain, self.n,
                              self.context)
        elif domain_descr == 'h':
            return type(self)(self.a_domain, self.b_domain, (self.c_mid_index, self.c_domain[1]), self.n,
                              self.context)
        else:
            assert False

    @staticmethod
    def quarter_mid(interval: tuple, mid: int, direction: int) -> int:
        """
        warn: this function work only with non-cycle indexes ( polygon should be parameterized from some extreme point,
        excluded for this task )
        :param interval:
        :param mid:
        :param direction:
        :return:
        """
        left, right = interval
        if direction == 1:
            if mid < right:
                return (mid + right) // 2
            else:
                assert False
        elif direction == -1:
            if left < mid:
                return (left + mid) // 2
            else:
                assert False
        else:
            assert False

    def __iadd__(self, domain_descr: str):
        # assert domain_descr == 'f' or domain_descr == 'g' or domain_descr == 'h'

        if domain_descr == 'f':
            self.a_mid_index = self.quarter_mid(self.a_domain, self.a_mid_index, 1)

        elif domain_descr == 'g':
            self.b_mid_index = self.quarter_mid(self.b_domain, self.b_mid_index, 1)

        elif domain_descr == 'h':
            self.c_mid_index = self.quarter_mid(self.c_domain, self.c_mid_index, 1)
        else:
            assert False

        return self

    def __isub__(self, domain_descr):
        # assert domain_descr == 'f' or domain_descr == 'g' or domain_descr == 'h'
        if domain_descr == 'f':
            self.a_mid_index = self.quarter_mid(self.a_domain, self.a_mid_index, -1)

        elif domain_descr == 'g':
            self.b_mid_index = self.quarter_mid(self.b_domain, self.b_mid_index, -1)

        elif domain_descr == 'h':
            self.c_mid_index = self.quarter_mid(self.c_domain, self.c_mid_index, -1)
        else:
            assert False

        return self

    def get_solution_and_score_from_found_domain(self, solution_points: tuple) -> tuple:
        cross_point_a, cross_point_b, cross_point_c = solution_points

        if cross_point_a is None or cross_point_b is None or cross_point_c is None:
            return None, -1e10
        corresponding_rc = axis_aligned_rectangle_from_2_point(cross_point_a, cross_point_c)

        dy = cross_point_a[1] - cross_point_b[1]
        dx = cross_point_b[0] - cross_point_c[0]

        return corresponding_rc, corresponding_rc.area()

    def predict_domain(self, h_0: int, h_1: int, h_T: int, f_T: int, g_T: int) -> tuple:
        domain = self
        ########################################################
        if (h_1 or h_T):
            h_T2 = 1
        elif (h_0 or (not h_T)):
            h_T2 = 0
        else:
            assert False

        key = (0 if f_T else 1, 0 if g_T else 1, h_T)
        if self.spaces_degraded.sum() == 0:
            # Tentative only in > 2 dims case
            ########################################################
            if key == (1, 1, 1):
                return -1, None
            elif key == (0, 0, 0):
                return 1, None
            ########################################################
        function_space, lo_up = domain.decode_domain_update(key)

        return function_space, lo_up

    def search_core(self, cyclic_list: CyclicList, stack: int = 0) -> tuple:
        domain = self
        # print(domain)
        context: ThreeFunctionsSearchContext = self.context

        while True:

            if self.debug:
                print('mid_index: %d[%d] %d[%d] %d[%d]' % (
                    domain.a_mid_index, domain.a_len, domain.b_mid_index, domain.b_len, domain.c_mid_index,
                    domain.c_len))

            if domain.a_len <= 2 and domain.b_len <= 2 and domain.c_len <= 2:
                return domain, stack

            if stack > context.MAX_ITERATIONS:
                return domain, stack

            def metric_1(cyclic_list_, a):
                return cyclic_list_.direction_from_point(a).slope() * self.sign_config['metric_1:slope_sign']

            def metric_2(cyclic_list_, a, b):
                return cyclic_list_.direction_from_2_points(a, b)

            m_a = metric_1(cyclic_list, domain.a_mid_index)
            m_b = metric_1(cyclic_list, domain.b_mid_index)
            m_c = metric_1(cyclic_list, domain.c_mid_index)

            m = -(m_c - m_b) / (m_a - m_b) * m_a

            m_ac = metric_2(cyclic_list, domain.a_mid_index, domain.c_mid_index)
            m_ac_slope = m_ac.slope() * self.sign_config['m_ac_slope:diagonal_sign']

            assert np.sign(m_a) == self.sign_config['control:m_a_sign']
            assert np.sign(m_b) == self.sign_config['control:m_b_sign']
            assert np.sign(m_c) == self.sign_config['control:m_c_sign']

            def score_f():
                a_y = (cyclic_list[domain.a_mid_index][1] + cyclic_list[domain.a_mid_index + 0][1]) / 2
                b_y = (cyclic_list[domain.b_mid_index][1] + cyclic_list[domain.b_mid_index + 0][1]) / 2
                score = (a_y - b_y) * self.sign_config['score_f:y_sign']
                if self.debug:
                    print(
                        '%d  f: %f,%f [%f]' % (
                            stack, a_y, b_y, abs(score)))
                return score

            def score_g():
                b_x = (cyclic_list[domain.b_mid_index][0] + cyclic_list[domain.b_mid_index + 0][0]) / 2
                c_x = (cyclic_list[domain.c_mid_index][0] + cyclic_list[domain.c_mid_index + 0][0]) / 2
                score = (b_x - c_x) * self.sign_config['score_g:x_sign']
                if self.debug:
                    print(
                        '%d  g: %f,%f [%f]' % (
                            stack, b_x, c_x, abs(score)))
                return score

            def score_h():
                score = (m - m_ac_slope) * self.sign_config['m_ac_slope:diagonal_sign']
                if self.debug:
                    print('%d  %f,%f [%f]' % (stack, m, m_ac_slope, abs(score)))
                return score

            h_3 = (-m_a >= -m_c)
            if self.sign_config['m_a__m_c:compare'] == 1:
                h_1 = (-m_a > m_ac_slope)
                h_0 = (m_ac_slope > -m_c)
            else:
                h_1 = (-m_a < m_ac_slope)
                h_0 = (m_ac_slope < -m_c)

            # h_0 = False
            # h_1 = False

            # #
            # if not h_3:
            #     if h_1 and h_0:
            #         new_domain2 = domain >> 'f'
            #         new_domain = new_domain2 << 'h'
            #         break
            #     if h_1:
            #         new_domain = domain >> 'f'
            #         break
            #     elif h_0:
            #         new_domain = domain << 'h'
            #         break

            f_T = 1 if score_f() > 0 else 0
            g_T = 1 if score_g() > 0 else 0
            h_T = 1 if score_h() > 0 else 0
            function_space, lo_up = self.predict_domain(h_0, h_1, h_T, f_T, g_T)

            if type(function_space) != int:
                # Apply non-consistent case
                if lo_up == 'up':
                    new_domain = domain >> function_space
                elif lo_up == 'lo':
                    new_domain = domain << function_space
                else:
                    assert False

                # Check previous tentative bounds reduction
                while len(context.tentative_stack) > 0:
                    _function_space_round_robin, key_tentative = context.tentative_stack.pop()
                    # previous is in current space, skip
                    if _function_space_round_robin == function_space:
                        continue

                    # apply tentative mid-point moving to up-quarter or lo-quarter
                    if key_tentative == 1:
                        new_domain = new_domain << _function_space_round_robin
                    elif key_tentative == -1:
                        new_domain = new_domain >> _function_space_round_robin

                    # context.tentative_stack.clear()
                    # break
                # self.context.diagonal_dump.append(m_ac)
                break

            # If case can't be resolved exactly try tentative

            spaces = self.spaces_filtered  # ['f', 'g', 'h']

            _function_space_round_robin = spaces[context.function_space_counter % len(spaces)]
            context.function_space_counter += 1
            key_stable = function_space
            context.tentative_stack.append((_function_space_round_robin, key_stable))
            if self.debug:
                print('tentative iter %s' % _function_space_round_robin)

            if key_stable == 1:
                domain -= _function_space_round_robin
            elif key_stable == -1:
                domain += _function_space_round_robin
            else:
                assert False

            if len(context.tentative_stack) > context.MAX_TENTATIVE_ITERATIONS:
                print('exit ..')
                return domain, stack

        return new_domain.search_core(cyclic_list, stack + 1)


class LiarSearch(PruneSearchTwoFunctions):
    """
    largest, inscribed, axis-aligned rectangle (2-corners part)
    """

    def __init__(self, f_domain: tuple, g_domain: tuple, n: int, context: GenericContext):
        super().__init__(f_domain, g_domain, n, context)

    def get_solution_and_score_from_found_domain(self, solution_points: tuple) -> tuple:
        cross_point_1, cross_point_2 = solution_points
        corresponding_rc = axis_aligned_rectangle_from_2_point(cross_point_1, cross_point_2)
        return corresponding_rc, corresponding_rc.area()

    def search_core(self,
                    cyclic_list: CyclicList,
                    stack: int = 0) -> tuple:
        domain = self

        if self.debug:
            print(domain)

        f_len, g_len = domain.f_len, domain.g_len
        if f_len <= 2 and g_len <= 2:
            return domain, stack

        g_mid_index = domain.g_mid_index
        f_mid_index = domain.f_mid_index

        if self.debug:
            print('mid_index: %d %d' % (g_mid_index, f_mid_index))

        def metric_1(c, a):
            return c.direction_from_point(a).slope()

        def metric_2(c, a, b):
            return c.direction_from_2_points(a, b)

        u_mid_index = f_mid_index
        l_mid_index = g_mid_index

        l_to_u_direction = metric_2(cyclic_list, l_mid_index, u_mid_index)
        u_dir_slope = metric_1(cyclic_list, u_mid_index)
        l_dir_slope = metric_1(cyclic_list, l_mid_index)

        self.context.diagonal_dump.append(l_to_u_direction)

        def score_f():
            score = (u_dir_slope - l_dir_slope)
            if self.debug:
                print('%d  P:%f,%f [%f]' % (stack, u_dir_slope, l_dir_slope, abs(score)))
            return score

        def score_g():
            score = (-l_to_u_direction.slope() - u_dir_slope)
            if self.debug:
                print('%d  D:%f,%f [%f]' % (stack, l_to_u_direction.slope(), u_dir_slope, abs(score)))
            return score

        compare_f = 1 if score_f() > 0 else 0
        compare_g = 1 if score_g() > 0 else 0

        if g_len < 2:
            new_domain = domain >> 'f' if compare_g else domain << 'f'
        elif f_len < 2:
            new_domain = domain >> 'g' if compare_f else domain << 'g'
        else:

            key = (compare_f, compare_g)
            function_space, lo_up = domain.decode_domain_update(domain.execution_inc_dec[key])

            if lo_up == 'up':
                new_domain = domain >> function_space
            elif lo_up == 'lo':
                new_domain = domain << function_space
            else:
                assert False

            # print(new_domain)
        return new_domain.search_core(cyclic_list, stack + 1)

    def refine_solution(self, cyclic_list: CyclicList, found_domain: tuple) -> tuple:
        i1, i2 = found_domain
        u_mid_dir = cyclic_list.direction_from_point(i1)
        l_mid_dir = cyclic_list.direction_from_point(i2)
        slope_target = -(u_mid_dir.slope() + l_mid_dir.slope()) / 2
        cross_point_1, cross_point_2, _ = refine_points_by_average_line_slope(cyclic_list, i1, i2,
                                                                              slope_target)
        return cross_point_1, cross_point_2


class LcpSearch(PruneSearchTwoFunctions):
    """
    Longest chord parallel to given direction
    """

    def __init__(self, f_domain: tuple, g_domain: tuple, n: int, context: TargetDirSearchContext):
        super().__init__(f_domain, g_domain, n, context)
        assert type(context) == TargetDirSearchContext
        self.target_dir = context.target_dir

    def get_solution_and_score_from_found_domain(self, solution_points: tuple) -> tuple:
        solution_dir = DirectionInformation(DirectionInformation.custom, solution_points[1], solution_points[0])
        score = solution_dir.dot_normed(self.target_dir)
        return solution_dir, score

    def refine_solution(self, cyclic_list: CyclicList, found_domain: tuple) -> tuple:
        i1, i2 = found_domain
        # alpha = -self.target_dir.angle() - np.pi / 2
        # rotation_matrix = np.array([[np.cos(alpha), -np.sin(alpha)],
        #                             [np.sin(alpha), np.cos(alpha)]])
        # transformed_cyclic_list = TransformedCyclicList(cyclic_list, rotation_matrix)

        # u_mid_dir = cyclic_list.direction_from_point(i1)
        # l_mid_dir = cyclic_list.direction_from_point(i2)

        slope_target = self.target_dir.slope()
        cross_point_1, cross_point_2, _ = refine_points_by_average_line_slope(cyclic_list, i1, i2,
                                                                              slope_target)

        # alpha = self.target_dir.angle() + np.pi / 2
        # rotation_matrix = np.array([[np.cos(alpha), -np.sin(alpha)],
        #                             [np.sin(alpha), np.cos(alpha)]])
        # transformed_cyclic_list = TransformedCyclicList([cross_point_1, cross_point_2], rotation_matrix)

        return cross_point_1, cross_point_2

    def search_core(self, cyclic_list: CyclicList, stack: int = 0) -> tuple:
        domain = self

        if self.debug:
            print(domain)

        f_len, g_len = domain.f_len, domain.g_len
        if f_len < 2 and g_len < 2:
            return domain, stack

        g_mid_index = domain.g_mid_index
        f_mid_index = domain.f_mid_index

        if self.debug:
            print('mid_index: %d %d' % (g_mid_index, f_mid_index))

        alpha = -self.target_dir.angle() - np.pi / 2
        rotation_matrix = np.array([[np.cos(alpha), -np.sin(alpha)],
                                    [np.sin(alpha), np.cos(alpha)]])

        transformed_cyclic_list = TransformedCyclicList(cyclic_list, rotation_matrix)

        def metric_1(c, a):
            return c.direction_from_point(a)

        def metric_2(c, a, b):
            return c.direction_from_2_points(a, b)

        u_mid_index = f_mid_index
        l_mid_index = g_mid_index

        # l_to_u_direction = metric_2(transformed_cyclic_list, l_mid_index, u_mid_index)
        u_dir = metric_1(cyclic_list, u_mid_index)
        l_dir = metric_1(cyclic_list, l_mid_index)

        self.context.diagonal_dump.append(metric_2(cyclic_list, l_mid_index, u_mid_index))

        def score_f():
            score = (transformed_cyclic_list[l_mid_index][0] - transformed_cyclic_list[u_mid_index][0])

            # print('%d  P:%f,%f [%f]' % (stack, u_dir_slope, l_dir_slope, abs(score)))
            return score

        def score_g():
            score = u_dir.cross_normed(l_dir)
            # print('%d  D:%f,%f [%f]' % (stack, l_to_u_direction.slope(), u_dir_slope, abs(score)))
            return score

        compare_f = 1 if score_f() > 0 else 0
        compare_g = 1 if score_g() > 0 else 0

        if g_len < 2:
            new_domain = domain >> 'f' if compare_f else domain << 'f'
        elif f_len < 2:
            new_domain = domain >> 'g' if compare_f else domain << 'g'
        else:

            key = (compare_f, compare_g)
            function_space, lo_up = domain.decode_domain_update(domain.execution_dec_inc[key])

            if lo_up == 'up':
                new_domain = domain >> function_space
            elif lo_up == 'lo':
                new_domain = domain << function_space
            else:
                assert False

            # print(new_domain)
        return new_domain.search_core(cyclic_list, stack + 1)


class ExtremePointsFinder:
    # Complexity 4*log(n)
    def __init__(self, cyclic_list: CyclicList):

        self.cyclic_list = cyclic_list

        left = cyclic_list.binary_search_min_x_vertex()
        right = cyclic_list.binary_search_max_x_vertex()
        top = cyclic_list.binary_search_max_y_vertex()
        bottom = cyclic_list.binary_search_min_y_vertex()

        self.extreme_points_original = {'x_': left,
                                        '_x': right,
                                        'y_': bottom,
                                        '_y': top}

        # clean-up NaN's
        i = left
        while True:
            s = cyclic_list.get_edge_slope(i)
            i += 1
            if np.isnan(s):
                left += 1
            else:
                break

        i = right
        while True:
            s = cyclic_list.get_edge_slope(i)
            i -= 1
            if np.isnan(s):
                right -= 1
            else:
                break

        self.extreme_points = {'x_': left,
                               '_x': right,
                               'y_': bottom,
                               '_y': top}

        self.n = len(cyclic_list)

    def get_two_function_configs(self, cleanup_nan_slopes=True, type__=LiarSearch, context_type=GenericContext,
                                 **kwargs) -> dict:
        extreme_points = self.extreme_points if cleanup_nan_slopes else self.extreme_points_original

        config = {'SCAN_TL_BR': type__((extreme_points['_y'], extreme_points['x_']),
                                       (extreme_points['y_'], extreme_points['_x']), self.n,
                                       context_type(**kwargs)),
                  'SCAN_TR_BL': type__((extreme_points['_x'], extreme_points['_y']),
                                       (extreme_points['x_'], extreme_points['y_']), self.n,
                                       context_type(**kwargs)),
                  'FULL': type__((extreme_points['x_'], extreme_points['_x']),
                                 (extreme_points['_x'], extreme_points['x_']), self.n,
                                 context_type(**kwargs))}
        return config

    def cleanup_indexes(self, intervals: dict):
        all_interval_indexes_corrected = {}
        # interval_slopes = []
        slope_eps = 1e-4

        for key, interval in intervals.items():
            indexes = get_indexes(self.cyclic_list, interval[0], interval[1])
            slopes = list(map(lambda x: self.cyclic_list.direction_from_point(x).slope(), indexes))
            assert len(indexes) == len(slopes)
            corrected_indexes = []
            for j in range(len(indexes)):
                if np.isfinite(slopes[j]) and abs(slopes[j]) > 0:
                    corrected_indexes.append(indexes[j])

            all_interval_indexes_corrected[key] = (corrected_indexes[0], corrected_indexes[-1])

        return all_interval_indexes_corrected

    def representative_point(self) -> list:
        # consider diagonal intersection for most complex cases
        x_ = self.cyclic_list[self.extreme_points_original['x_']][0] + \
             self.cyclic_list[self.extreme_points_original['_x']][0]
        y_ = self.cyclic_list[self.extreme_points_original['y_']][1] + \
             self.cyclic_list[self.extreme_points_original['_y']][1]
        return [x_ / 2, y_ / 2]

    def get_three_function_configs(self, cleanup_nan_slopes=True, type__=TentativeSearchThreeFunctions,
                                   context_type=ThreeFunctionsSearchContext,
                                   **kwargs) -> dict:
        extreme_points = self.extreme_points_original

        intervals_set = {'1': (extreme_points['_y'], extreme_points['x_']),
                         '2': (extreme_points['_x'], extreme_points['_y']),
                         '3': (extreme_points['y_'], extreme_points['_x']),
                         '4': (extreme_points['x_'], extreme_points['y_'])}
        ##############################################################################################
        intervals = {'a': intervals_set['1'],
                     'b': intervals_set['2'],
                     'c': intervals_set['3']}

        intervals = self.cleanup_indexes(intervals)

        config = {'SCAN_LEADING_TL': type__(intervals['a'], intervals['b'], intervals['c'],
                                            self.n,
                                            context_type('SCAN_LEADING_TL', **kwargs))}

        intervals = {'a': intervals_set['4'],
                     'b': intervals_set['3'],
                     'c': intervals_set['2']}

        intervals = self.cleanup_indexes(intervals)

        config['SCAN_LEADING_BL'] = type__(intervals['a'], intervals['b'], intervals['c'],
                                           self.n,
                                           context_type('SCAN_LEADING_BL', **kwargs))

        intervals = {'a': intervals_set['3'],
                     'b': intervals_set['4'],
                     'c': intervals_set['1']}

        intervals = self.cleanup_indexes(intervals)

        config['SCAN_LEADING_BR'] = type__(intervals['a'], intervals['b'], intervals['c'],
                                           self.n,
                                           context_type('SCAN_LEADING_BR', **kwargs))

        intervals = {'a': intervals_set['2'],
                     'b': intervals_set['1'],
                     'c': intervals_set['4']}

        intervals = self.cleanup_indexes(intervals)

        config['SCAN_LEADING_TR'] = type__(intervals['a'], intervals['b'], intervals['c'],
                                           self.n,
                                           context_type('SCAN_LEADING_TR', **kwargs))

        return config

# class PruneSearchOneFunction(GenericExecution):
#
#     def __init__(self, f_domain: tuple, n: int, target_dir: DirectionInformation, diagonal_dump: list = None):
#         super().__init__(n, diagonal_dump)
#         assert type(f_domain) == tuple
#         self.f_domain = f_domain
#         self.f_len = self._get_interval_len(self.f_domain[1], self.f_domain[0])
#         self.f_mid_index = (self.f_domain[0] + self.f_len // 2) % self.n
#         self.target_dir = target_dir
#         if diagonal_dump is not None:
#             self.diagonal_dump.append(target_dir)
#
#     def search_core(self, cyclic_list: CyclicList, stack: int = 0) -> tuple:
#         domain = self
#         print(domain)
#
#         f_len = domain.f_len
#         if f_len <= 2:
#             return domain, stack
#
#         f_mid_index = domain.f_mid_index
#
#         print('mid_index: %d' % f_mid_index)
#         score_ = cyclic_list.direction_from_point(f_mid_index).dot_normed(self.target_dir)
#
#         if score_ >= 0:
#             new_domain = domain << 'f'
#
#         else:
#             new_domain = domain >> 'f'
#
#         return new_domain.search_core(cyclic_list, stack + 1)
#
#     def __repr__(self):
#         return 'f: {}'.format(self.f_domain)
#
#     def __str__(self) -> str:
#         return 'f: {}'.format(self.f_domain)
#
#     def __lshift__(self, domain_descr: str):
#         # assert domain_descr == 'f'
#         if domain_descr == 'f':
#             return PruneSearchOneFunction((self.f_domain[0], self.f_mid_index), self.n, self.target_dir)
#         else:
#             assert False
#
#     def __rshift__(self, domain_descr: str):
#         # assert domain_descr == 'f'
#         if domain_descr == 'f':
#             return PruneSearchOneFunction((self.f_mid_index, self.f_domain[1]), self.n, self.target_dir)
#         else:
#             assert False
