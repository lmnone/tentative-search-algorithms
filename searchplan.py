import itertools

import cv2
import numpy as np
from sortedcontainers import SortedList

from axisalignedrectangle import AxisAlignedRectangle
from binary_search_execution import GenericExecution, TentativeSearchThreeFunctions, PruneSearchTwoFunctions
from cyclic_list import CyclicList
from direction_information import DirectionInformation


class SolutionInfo:

    def __init__(self, cyclic_list: CyclicList, direct_solution_indexes: tuple, refined_solution: tuple = None):
        self.cyclic_list = cyclic_list

        assert type(direct_solution_indexes) == tuple

        assert len(direct_solution_indexes) == 2 or len(direct_solution_indexes) == 3

        if refined_solution is not None:
            assert type(refined_solution) == tuple
            assert len(refined_solution) == 2 or len(refined_solution) == 3
            assert len(refined_solution) == len(direct_solution_indexes)

        self.direct_solution_indexes = direct_solution_indexes
        self.refined_solution = refined_solution

    @property
    def best_points_available(self):
        if self.refined_solution is not None:
            return self.refined_solution
        else:
            return tuple(map(lambda x: self.cyclic_list[x], self.direct_solution_indexes))

    def __repr__(self):
        if len(self.direct_solution_indexes) == 2:
            if self.refined_solution is None:
                return '2 points: {},{}'.format(self.direct_solution_indexes[0], self.direct_solution_indexes[1])
            else:
                return '2 points: {},{} => {},{}'.format(self.direct_solution_indexes[0],
                                                         self.direct_solution_indexes[1],
                                                         self.refined_solution[0], self.refined_solution[1])
        elif len(self.direct_solution_indexes) == 3:
            if self.refined_solution is None:
                return '3 points: {},{},{}'.format(self.direct_solution_indexes[0], self.direct_solution_indexes[1],
                                                   self.direct_solution_indexes[2])
            else:
                return '3 points: {},{},{} => {},{},{}'.format(self.direct_solution_indexes[0],
                                                               self.direct_solution_indexes[1],
                                                               self.direct_solution_indexes[2],
                                                               self.refined_solution[0], self.refined_solution[1],
                                                               self.refined_solution[2])


class SearchPlan:

    def __init__(self, cyclic_list: CyclicList, kind: str, execution: GenericExecution, refine=False):
        assert type(kind) == str

        self.refine = refine
        self.scores = None
        self.iterations = None
        self.kind = kind
        self.cyclic_list = cyclic_list
        self.execution = execution

    @property
    def domain_points(self) -> list:
        # only for debug
        domain_indexes = self.execution.retrieve_indexes(self.cyclic_list)
        domain_points_ = []
        for domain_index in domain_indexes:
            domain_points_.append(list(map(lambda i__: self.cyclic_list[i__], domain_index)))
        return domain_points_

    def __repr__(self):
        return '@PLAN <{}> MAX:{}'.format(self.kind, np.round(self.score(), 2) if self.scores is not None else '')

    def descr(self):
        return '{}'.format(
            self.kind.lower().replace(' ', '').replace('-', '_').replace('+', '_'))

    def apply_plan(self):
        """
        Find the largest, inscribed, axis-aligned rectangle using tentative search algorithm
        ( selective index part )

        """
        found_domain, self.iterations = self.execution.search_core(self.cyclic_list)

        if not isinstance(found_domain, PruneSearchTwoFunctions) and not isinstance(found_domain,
                                                                                    TentativeSearchThreeFunctions):
            assert False

        solution_info = None
        self.scores = SortedList(key=lambda x: -x[0])
        if isinstance(found_domain, PruneSearchTwoFunctions):
            # 2 functions search
            a1 = list(found_domain.f_domain)
            b2 = list(found_domain.g_domain)

            for pair in itertools.product(a1, b2):
                i1, i2 = pair
                if i1 is None or i2 is None:
                    continue

                # print((i1, i2))
                if self.refine:
                    cross_point_1, cross_point_2 = self.execution.refine_solution(self.cyclic_list,
                                                                                  found_domain=(i1, i2))

                    if np.any(list(map(lambda x: x is None, [cross_point_1, cross_point_2]))):
                        continue

                    solution_info = SolutionInfo(self.cyclic_list,
                                                 direct_solution_indexes=(i1, i2),
                                                 refined_solution=(cross_point_1, cross_point_2))
                else:
                    solution_info = SolutionInfo(self.cyclic_list,
                                                 direct_solution_indexes=(i1, i2))
                # universal part

                solution_object, solution_score = self.execution.get_solution_and_score_from_found_domain(
                    solution_info.best_points_available)
                self.scores.add(
                    (solution_score, solution_info, solution_object))

        else:
            # 3 function search
            a1 = list(found_domain.a_domain)
            b2 = list(found_domain.b_domain)
            c3 = list(found_domain.c_domain)

            # print((a1, b2, c3))

            for triple in itertools.product(a1, b2, c3, [-1, 1], [-1, 1], [-1, 1]):
                i1, i2, i3, shift_a, shift_b, shift_c = triple
                if i1 is None or i2 is None or i3 is None:
                    continue

                # print((i1, i2, i3))

                if self.refine:
                    cross_point_1, cross_point_2, cross_point_3 = self.execution.refine_solution(self.cyclic_list,
                                                                                                 found_domain=(
                                                                                                     i1, i2, i3),
                                                                                                 shifts=(
                                                                                                     shift_a, shift_b,
                                                                                                     shift_c))

                    if np.any(list(map(lambda x: x is None, [cross_point_1, cross_point_2, cross_point_3]))):
                        continue

                    solution_info = SolutionInfo(self.cyclic_list,
                                                 direct_solution_indexes=(i1, i2, i3),
                                                 refined_solution=(cross_point_1, cross_point_2, cross_point_3))

                else:
                    solution_info = SolutionInfo(self.cyclic_list,
                                                 direct_solution_indexes=(i1, i2, i3))
                # universal part

                solution_object, solution_score = self.execution.get_solution_and_score_from_found_domain(
                    solution_info.best_points_available)
                self.scores.add(
                    (solution_score, solution_info, solution_object))

        best_score = self.scores[0]
        solution_score, solution_info, solution_object = best_score

        return solution_object, solution_info

    def score(self) -> float:
        return self.scores[0][0]

    def optimal_solution(self) -> object:
        best_score = self.scores[0]
        _, __, solution_object = best_score
        return solution_object

    def optimal_solution_info(self) -> SolutionInfo:
        best_score = self.scores[0]
        _, solution_info, __ = best_score
        return solution_info

    def draw_found_domain_points(self, canvas: np.array):
        solution_object = self.optimal_solution()
        c = self.cyclic_list
        for p_ in solution_object:
            cv2.circle(canvas, c[p_[0]], 5, 255, 2)
            cv2.circle(canvas, c[p_[1]], 5, 255, 2)

    def draw_cross_points(self, canvas: np.array):
        if self.execution.context.dump_type == 'ThreeFunctionsSearchContext':
            for p1 in self.execution.context.cross_point_with_borders:
                color_cross_points = (0, 0, 255)
                cv2.circle(canvas, np.int0(p1), 2, color_cross_points, 1)
        else:
            cross_point_1, cross_point_2, cross_point_3 = self.optimal_solution_info().refined_solution

            color_cross_points = (0, 0, 255)
            cv2.circle(canvas, np.int0(cross_point_1), 10, color_cross_points, 2)
            cv2.circle(canvas, np.int0(cross_point_2), 10, color_cross_points, 2)
            cv2.circle(canvas, np.int0(cross_point_3), 10, color_cross_points, 2)

    def draw_polylines(self, canvas: np.array, use_markers=False):
        color_polylines = [
            (175, 175, 175), (125, 125, 125), (75, 75, 75)]

        for i in range(0, len(self.domain_points)):
            if use_markers:
                for point in self.domain_points[i]:
                    cv2.drawMarker(canvas, np.round(point).astype('int'), color_polylines[i],
                                   markerType=cv2.MARKER_CROSS,
                                   markerSize=5, thickness=1, line_type=cv2.LINE_AA)
            else:
                cv2.polylines(canvas, [np.round(np.array(self.domain_points[i])).astype('int')], False,
                              color_polylines[i],
                              thickness=2 + i)

    def draw_optimal_solution(self, canvas: np.array):
        if self.execution.context.dump_type == 'diagonal_search':
            optimal_diagonal = self.optimal_solution()
            assert type(optimal_diagonal) == DirectionInformation
            diagonal_points = np.round(np.array([optimal_diagonal.pt1, optimal_diagonal.pt2])).astype('int')
            cv2.arrowedLine(canvas, diagonal_points[0], diagonal_points[1],
                            (0, 210, 195), thickness=3)
        else:

            optimal_rc = self.optimal_solution()
            assert type(optimal_rc) == AxisAlignedRectangle
            color_optimal_rc = (0, 250, 0)
            cv2.rectangle(canvas, np.int0(optimal_rc.bl), np.int0(optimal_rc.tr), color_optimal_rc, 1)

    def draw_refine_boxes(self, canvas: np.array):
        for box in self.execution.context.refine_boxes:
            assert type(box) == AxisAlignedRectangle
            color_optimal_rc = (0, 250, 0)
            cv2.rectangle(canvas, np.int0(box.bl), np.int0(box.tr), color_optimal_rc, 1)

    def draw_diagonal_dump(self, canvas: np.array, draw_numbers: bool = False):
        stack = 0
        color_diagonal = (255, 255, 0)
        color_diagonal_0 = (32, 32, 187)
        color_last = (0, 255, 0)
        color_text = (255, 255, 255)

        for dir__ in self.execution.context.diagonal_dump:
            if draw_numbers:
                cv2.putText(canvas, str(stack), np.int0(dir__.pt1), cv2.FONT_HERSHEY_PLAIN, thickness=1, fontScale=1,
                            color=color_text)
            diagonal_points = np.round(np.array([dir__.pt1, dir__.pt2])).astype('int')

            if self.execution.context.dump_type == 'diagonal_search':
                if stack == 0:
                    cv2.arrowedLine(canvas, diagonal_points[0], diagonal_points[1], color_diagonal_0, thickness=6)
                elif stack == len(self.execution.context.diagonal_dump) - 1:
                    cv2.arrowedLine(canvas, diagonal_points[0], diagonal_points[1],
                                    color_last, thickness=1)
                # else:
                #     cv2.arrowedLine(canvas, diagonal_points[0], diagonal_points[1],
                #                     color_diagonal, thickness=1)
            else:
                if stack == len(self.execution.context.diagonal_dump) - 1:
                    cv2.arrowedLine(canvas, diagonal_points[0], diagonal_points[1],
                                    color_last, thickness=2)
                else:
                    cv2.arrowedLine(canvas, diagonal_points[0], diagonal_points[1],
                                    color_diagonal, thickness=1)

            stack += 1

    def refine_strength(self) -> str:
        optimal_solution_info = self.optimal_solution_info()
        i1, i2 = optimal_solution_info.direct_solution_indexes
        cross_point_1, cross_point_2 = optimal_solution_info.refined_solution

        _, original_score = self.execution.get_solution_and_score_from_found_domain(
            (self.cyclic_list[i1], self.cyclic_list[i2]))
        __, refined_score = self.execution.get_solution_and_score_from_found_domain(
            (cross_point_1, cross_point_2))

        descr = '%s => %s' % (original_score, refined_score)
        return descr
