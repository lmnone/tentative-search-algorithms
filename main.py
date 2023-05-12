import os
import sys

import cv2
import numpy as np
from convex_poly import get_maximal_rectangle
from computational_geometry import compute_largest_inscribed_isothetic_rectangle, \
    compute_longest_chord_parallel_to_given_direction
from cyclic_list import CyclicList
import pickle
from prettytable import PrettyTable
import tqdm
from triangles import draw_curve


def evaluation_compute_largest_inscribed_isothetic_rectangle__lp(path: str, dump_path: str):
    """

    :type dump_path: object
    """
    I = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    # print((I.shape, I.dtype))
    contours_orig, hierarchy__ = cv2.findContours(I, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    # print(len(contours_orig))

    if len(contours_orig) == 0:
        raise Exception('can\'t compute LIIR with no contours')
    if len(contours_orig) > 1:
        raise Exception('can\'t compute LIIR in more than one contour')

    canvas = np.zeros_like(I).astype('uint8')
    convex_polygon = cv2.convexHull(contours_orig[0])
    cv2.drawContours(canvas, [convex_polygon], -1, 128, 2, cv2.LINE_AA)
    convex_polygon = convex_polygon.reshape(-1, 2)

    filename__ = os.path.basename(path).split('.')[0]
    with open(os.path.join(dump_path, filename__) + '.pickle', 'wb') as handle:
        pickle.dump(convex_polygon.tolist(), handle, protocol=pickle.HIGHEST_PROTOCOL)

    bl, tr, area__ = get_maximal_rectangle(convex_polygon.tolist())
    cv2.rectangle(canvas, list(bl.astype('int32')), list(tr.astype('int32')), 250, 1)

    cv2.imwrite(os.path.join(dump_path, filename__) + '.jpg', canvas)
    # print('|%s|%d|%f' % (filename__, len(convex_polygon.tolist()), area))
    return len(convex_polygon.tolist()), area__, convex_polygon.tolist()


def test(a: list):
    a.append(0)


if __name__ == '__main__':
    image_samples_folder = sys.argv[1]
    dump_path = sys.argv[2]

    t = PrettyTable(['filename', 'edges', 'area(lp)', 'area', 'iterations', 'optimal plan'])

    for filename in tqdm.tqdm(sorted(os.listdir(image_samples_folder))):
        filepath = os.path.join(image_samples_folder, filename)
        # if not (filename == 'sample_93.png'):
        #     continue
        n_edges, area, convex_poly = evaluation_compute_largest_inscribed_isothetic_rectangle__lp(filepath, dump_path)
        c = CyclicList(convex_poly)
        area_tentative, iterations, plan_config_dict_liir, optimal_plan_descr = compute_largest_inscribed_isothetic_rectangle(
            c, refine=True)

        if area_tentative is None:
            continue

        basename_hint = os.path.basename(filename).replace('png', '')
        canvas = np.zeros([512, 512, 3], dtype=np.uint8)
        filepath = os.path.join(dump_path, basename_hint + '_' + optimal_plan_descr) + '.jpg'

        draw_curve(c, canvas)
        plan_config_dict_liir[optimal_plan_descr].draw_optimal_solution(canvas)

        cv2.imwrite(filepath, canvas)

        # end dump part

        t.add_row(
            [filename, n_edges, np.round(area, 2), np.round(area_tentative.area(), 2), iterations, optimal_plan_descr])

    print(t)
