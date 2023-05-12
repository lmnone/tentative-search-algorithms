import numpy as np
import cvxpy

from shapely.geometry import Polygon


def two_pts_to_line(pt1, pt2):
    """
    Create a line from two points in form of
    a1(x) + a2(y) = b
    """

    a1 = float(pt1[1] - pt2[1])
    a2 = float(pt2[0] - pt1[0])
    b = float(pt2[0] * pt1[1] - pt1[0] * pt2[1])

    return a1, a2, b


def pts_to_leq(coords):
    """
    Converts a set of points to form Ax = b, but since
    x is of length 2 this is like A1(x1) + A2(x2) = B.
    returns A1, A2, B
    """

    A1 = []
    A2 = []
    B = []
    for i in range(len(coords) - 1):
        pt1 = coords[i]
        pt2 = coords[i + 1]
        a1, a2, b = two_pts_to_line(pt1, pt2)
        A1.append(a1)
        A2.append(a2)
        B.append(b)
    return A1, A2, B


def get_maximal_rectangle(coordinates__):
    """
    Find the largest, inscribed, axis-aligned rectangle.
    :param coordinates__:
        A list of [x, y] pairs describing a closed, convex polygon.
    """
    eps = 1e-5
    coordinates = coordinates__
    coordinates.append(coordinates__[0])
    coordinates = np.array(coordinates)
    xy_view_minimum = np.min(coordinates, axis=0)
    xy_view_maximum = np.max(coordinates, axis=0)

    x_range = xy_view_maximum[0] - xy_view_minimum[0]
    y_range = xy_view_maximum[1] - xy_view_minimum[1]

    scale = np.array([x_range, y_range])
    sc_coordinates = (coordinates - xy_view_minimum)/ scale

    poly = Polygon(sc_coordinates)
    inside_pt = (poly.representative_point().x,
                 poly.representative_point().y)

    A1, A2, B = pts_to_leq(sc_coordinates)

    bl = cvxpy.Variable(2, pos=True)
    tr = cvxpy.Variable(2, pos=True)
    br = cvxpy.Variable(2, pos=True)
    tl = cvxpy.Variable(2, pos=True)

    obj = cvxpy.Maximize(cvxpy.log(tr[0] - bl[0]) + cvxpy.log(tr[1] - bl[1]))
    constraints = [
        bl[0] == tl[0],
        br[0] == tr[0],
        tl[1] == tr[1],
        bl[1] == br[1],
        tr[0] >= bl[0] + eps,
        tr[1] >= bl[1] + eps
    ]

    for i in range(len(B)):
        if inside_pt[0] * A1[i] + inside_pt[1] * A2[i] <= B[i]:
            constraints.append(bl[0] * A1[i] + bl[1] * A2[i] <= B[i] - eps)
            constraints.append(tr[0] * A1[i] + tr[1] * A2[i] <= B[i] - eps)
            constraints.append(br[0] * A1[i] + br[1] * A2[i] <= B[i] - eps)
            constraints.append(tl[0] * A1[i] + tl[1] * A2[i] <= B[i] - eps)

        else:
            constraints.append(bl[0] * A1[i] + bl[1] * A2[i] >= B[i] + eps)
            constraints.append(tr[0] * A1[i] + tr[1] * A2[i] >= B[i] + eps)
            constraints.append(br[0] * A1[i] + br[1] * A2[i] >= B[i] + eps)
            constraints.append(tl[0] * A1[i] + tl[1] * A2[i] >= B[i] + eps)

    prob = cvxpy.Problem(obj, constraints)
    prob.solve(verbose=False, solver=cvxpy.SCS)

    bottom_left = np.array(bl.value).T * scale + xy_view_minimum
    top_right = np.array(tr.value).T * scale + xy_view_minimum
    # print('result area: %f' % (np.prod(top_right - bottom_left)))
    area = (np.prod(top_right - bottom_left))
    return bottom_left, top_right, area
