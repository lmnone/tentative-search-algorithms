import numpy as np
import math


def two_pts_to_line(pt1, pt2, normalize_sign=False):
    """
    Create a line from two points in form of
    a1(x) + a2(y) = b
    """

    a1 = float(pt1[1] - pt2[1])
    a2 = float(pt2[0] - pt1[0])
    b = float(pt2[0] * pt1[1] - pt1[0] * pt2[1])

    if normalize_sign:
        if np.sign(a2) == -1:
            a1 *= -1
            a2 *= -1
            b *= -1

    return a1, a2, b


def point_of_segment_intersection(a0: list, a1: list, b0: list, b1: list):
    """
        Actually it's line intersection
    :param a0: x,y coordinate of first point on segment_A
    :param a1: x,y coordinate of second point on segment_A
    :param b0: x,y coordinate of first point on segment_B
    :param b1: x,y coordinate of second point on segment_B
    :return: Line intersection coordinates (x,y)
    """
    A0 = a0 + [1]
    B0 = b0 + [1]
    A1 = a1 + [1]
    B1 = b1 + [1]
    X = np.cross(np.cross(A0, A1), np.cross(B0, B1))
    assert abs(X[2]) > 0
    x = np.array([X[0], X[1]]) / X[2]
    return x.tolist()


class DirectionInformation:
    x = 1
    y = 2
    # i.e. line
    custom = 3

    def __init__(self, pc, pt1=None, pt2=None):
        if type(pc) == int:
            self.pc = pc
        elif type(pc) == str:
            self.pc = {'x': self.x, 'y': self.y, 'custom': self.custom}[pc]

        if pc == self.custom:
            assert pt1 is not None
            assert pt2 is not None
            self.a1, self.a2, self.b = two_pts_to_line(pt1, pt2)
            self.pt1 = pt1
            self.pt2 = pt2

    def query_dimension_order(self):
        if self.pc == self.x:
            return 0
        elif self.pc == self.y:
            return 1
        elif self.pc == self.custom:
            return -1
        else:
            assert False

    def query_opposite_dimension_order(self):
        if self.pc == self.x:
            return 1
        elif self.pc == self.y:
            return 0
        elif self.pc == self.custom:
            return -1
        else:
            assert False

    def __repr__(self):
        if self.pc == self.x:
            return 'x'
        elif self.pc == self.y:
            return 'y'
        elif self.pc == self.custom:
            def sign_to_str(var) -> str:
                if var > 0:
                    return '+'
                else:
                    return '-'

            return '@({}*x {} {}*y {} {} = 0)'.format(round(self.a1, 2), sign_to_str(self.a2),
                                                      abs(round(self.a2, 2)), sign_to_str(self.b),
                                                      abs(round(self.b, 2)))
        else:
            assert False

    def __eq__(self, other):
        if type(other) == int:
            return self.pc == other
        else:
            return self.pc == other.pc

    def is_same_halfplane(self, p1: list, p2: list) -> bool:
        return np.sign(p1[0] * self.a1 + p1[1] * self.a2 - self.b) == np.sign(
            p2[0] * self.a1 + p2[1] * self.a2 - self.b)

    def is_on_line(self, p1: list, eps: float = 1e-6) -> bool:
        return abs(p1[0] * self.a1 + p1[1] * self.a2 - self.b) < eps

    def slope(self, nan_eps=1e-6):
        if abs(self.a2) < nan_eps:
            # assert False
            return np.nan
        return -self.a1 / self.a2

    def angle(self) -> float:
        return math.atan2(self.pt2[1] - self.pt1[1], self.pt2[0] - self.pt1[0])

    def length(self) -> float:
        return math.sqrt((self.pt2[1] - self.pt1[1]) ** 2 + (self.pt2[0] - self.pt1[0]) ** 2)

    def dot_normed(self, other):
        n1 = np.array([self.a1, self.a2])
        n2 = np.array([other.a1, other.a2])
        angle_1 = np.dot(n1 / np.linalg.norm(n1), n2 / np.linalg.norm(n2))
        return angle_1

    def dot(self, other):
        n1 = np.array([self.a1, self.a2])
        n2 = np.array([other.a1, other.a2])
        angle_1 = np.dot(n1, n2)
        return angle_1

    def cross_normed(self, other):
        n1 = np.array([self.a1, self.a2])
        n2 = np.array([other.a1, other.a2])
        angle_1 = np.cross(n1 / np.linalg.norm(n1), n2 / np.linalg.norm(n2))
        return angle_1
