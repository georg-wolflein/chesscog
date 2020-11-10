from sklearn.metrics.pairwise import pairwise_distances
from sklearn.cluster import AgglomerativeClustering, DBSCAN
import cv2
import numpy as np
import typing

from chesscog.utils.io import URI
from chesscog.utils.coordinates import from_homogenous_coordinates, to_homogenous_coordinates
from chesscog.utils import sort_corner_points
from chesscog.utils.config import CfgNode as CN
from .visualise import draw_lines


def find_corners(cfg: CN, img: np.ndarray) -> np.ndarray:
    edges = detect_edges(cfg, img)
    lines = detect_lines(cfg, edges)
    all_horizontal_lines, all_vertical_lines = cluster_horizontal_and_vertical_lines(
        lines)

    horizontal_lines = eliminate_similar_lines(
        all_horizontal_lines, all_vertical_lines)
    vertical_lines = eliminate_similar_lines(
        all_vertical_lines, all_horizontal_lines)

    intersection_points = get_intersection_points(horizontal_lines,
                                                  vertical_lines)

    best_num_inliers = 0
    best_configuration = None
    iterations = 0
    while iterations < 30 or best_num_inliers < 30:
        row1, row2 = _choose_from_range(len(horizontal_lines))
        col1, col2 = _choose_from_range(len(vertical_lines))
        transformation_matrix = compute_homography(intersection_points,
                                                   row1, row2, col1, col2)
        warped_points = warp_points(transformation_matrix, intersection_points)
        warped_points, *_ = configuration =\
            discard_outliers(cfg, warped_points, intersection_points)
        num_inliers = np.prod(warped_points.shape[:-1])
        if num_inliers > best_num_inliers:
            best_num_inliers = num_inliers
            best_configuration = configuration
        iterations += 1
        if iterations > 100:
            return None

    # Retrieve best configuration
    warped_points, intersection_points, horizontal_scale, vertical_scale = best_configuration

    # Recompute transformation matrix based on all inliers
    col_xs, row_ys, quantized_points = quantize_points(
        warped_points, horizontal_scale, vertical_scale)
    transformation_matrix = compute_transformation_matrix(intersection_points,
                                                          quantized_points)

    # Get board boundaries
    inverse_transformation_matrix = np.linalg.inv(transformation_matrix)
    xmin, xmax = compute_vertical_borders(cfg,                                          col_xs, row_ys, edges,
                                          horizontal_scale, vertical_scale, inverse_transformation_matrix)
    ymin, ymax = compute_horizontal_borders(cfg,                                            col_xs, row_ys, edges,
                                            horizontal_scale, vertical_scale, inverse_transformation_matrix)

    # Transform boundaries to image space
    corners = np.array([[xmin, ymin],
                        [xmax, ymin],
                        [xmax, ymax],
                        [xmin, ymax]]).astype(np.float)
    scale = np.array([horizontal_scale, vertical_scale])
    corners = corners / scale
    img_corners = warp_points(inverse_transformation_matrix, corners)
    return sort_corner_points(img_corners)


def detect_edges(cfg: CN, img: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray,
                      cfg.EDGE_DETECTION.LOW_THRESHOLD,
                      cfg.EDGE_DETECTION.HIGH_THRESHOLD,
                      cfg.EDGE_DETECTION.APERTURE)
    return edges


def detect_lines(cfg: CN, edges: np.ndarray) -> np.ndarray:
    # array of [rho, theta]
    lines = cv2.HoughLines(edges, 1, np.pi/360, cfg.LINE_DETECTION.THRESHOLD)
    lines = lines.squeeze(axis=-2)
    return lines


def _fix_negative_rho_in_hesse_normal_form(lines: np.ndarray) -> np.ndarray:
    lines = lines.copy()
    neg_rho_mask = lines[..., 0] < 0
    lines[neg_rho_mask, 0] = - \
        lines[neg_rho_mask, 0]
    lines[neg_rho_mask, 1] =  \
        lines[neg_rho_mask, 1] - np.pi
    return lines


def _absolute_angle_difference(x, y):
    diff = np.mod(np.abs(x - y), 2*np.pi)
    return np.min(np.stack([diff, np.pi - diff], axis=-1), axis=-1)


def cluster_horizontal_and_vertical_lines(lines: np.ndarray):
    thetas = lines[..., 1].reshape(-1, 1)
    distance_matrix = pairwise_distances(
        thetas, thetas, metric=_absolute_angle_difference)
    agg = AgglomerativeClustering(n_clusters=2, affinity="precomputed",
                                  linkage="average")
    clusters = agg.fit_predict(distance_matrix)

    angle_with_y_axis = _absolute_angle_difference(thetas, 0.)
    if angle_with_y_axis[clusters == 0].mean() > angle_with_y_axis[clusters == 1].mean():
        hcluster, vcluster = 0, 1
    else:
        hcluster, vcluster = 1, 0

    horizontal_lines = lines[clusters == hcluster]
    vertical_lines = lines[clusters == vcluster]

    horizontal_lines = _fix_negative_rho_in_hesse_normal_form(horizontal_lines)
    vertical_lines = _fix_negative_rho_in_hesse_normal_form(vertical_lines)

    return horizontal_lines, vertical_lines


def eliminate_similar_lines(lines: np.ndarray, perpendicular_lines: np.ndarray) -> np.ndarray:
    perp_rho, perp_theta = perpendicular_lines.mean(axis=0)
    rho, theta = np.moveaxis(lines, -1, 0)
    intersection_points = get_intersection_point(
        rho, theta, perp_rho, perp_theta)
    intersection_points = np.stack(intersection_points, axis=-1)

    clustering = DBSCAN(eps=8, min_samples=1).fit(intersection_points)

    filtered_lines = []
    for c in range(clustering.labels_.max() + 1):
        lines_in_cluster = lines[clustering.labels_ == c]
        rho = lines_in_cluster[..., 0]
        median = np.argsort(rho)[len(rho)//2]
        filtered_lines.append(lines_in_cluster[median])
    return np.stack(filtered_lines)


def get_intersection_point(rho1, theta1, rho2, theta2):
    # rho1 = x cos(theta1) + y sin(theta1)
    # rho2 = x cos(theta2) + y sin(theta2)
    cos_t1 = np.cos(theta1)
    cos_t2 = np.cos(theta2)
    sin_t1 = np.sin(theta1)
    sin_t2 = np.sin(theta2)
    x = (sin_t1 * rho2 - sin_t2 * rho1) / (cos_t2 * sin_t1 - cos_t1 * sin_t2)
    y = (cos_t1 * rho2 - cos_t2 * rho1) / (sin_t2 * cos_t1 - sin_t1 * cos_t2)
    return x, y


def _choose_from_range(upper_bound: int, n: int = 2):
    return np.sort(np.random.choice(np.arange(upper_bound), (n,), replace=False), axis=-1)


def get_intersection_points(horizontal_lines: np.ndarray, vertical_lines: np.ndarray) -> np.ndarray:
    rho1, theta1 = np.moveaxis(horizontal_lines, -1, 0)
    rho2, theta2 = np.moveaxis(vertical_lines, -1, 0)

    rho1, rho2 = np.meshgrid(rho1, rho2, indexing="ij")
    theta1, theta2 = np.meshgrid(theta1, theta2, indexing="ij")
    intersection_points = get_intersection_point(rho1, theta1, rho2, theta2)
    intersection_points = np.stack(intersection_points, axis=-1)
    return intersection_points


def compute_transformation_matrix(src_points: np.ndarray, dst_points: np.ndarray) -> np.ndarray:
    transformation_matrix, _ = cv2.findHomography(src_points.reshape(-1, 2),
                                                  dst_points.reshape(-1, 2))
    return transformation_matrix


def compute_homography(intersection_points: np.ndarray, row1: int, row2: int, col1: int, col2: int):
    p1 = intersection_points[row1, col1]  # top left
    p2 = intersection_points[row1, col2]  # top right
    p3 = intersection_points[row2, col2]  # bottom right
    p4 = intersection_points[row2, col1]  # bottom left

    src_points = np.stack([p1, p2, p3, p4])
    dst_points = np.array([[0, 0],  # top left
                           [1, 0],  # top right
                           [1, 1],  # bottom right
                           [0, 1]])  # bottom left
    return compute_transformation_matrix(src_points, dst_points)


def warp_points(transformation_matrix: np.ndarray, points: np.ndarray) -> np.ndarray:
    points = to_homogenous_coordinates(points)
    warped_points = points @ transformation_matrix.T
    return from_homogenous_coordinates(warped_points)


def find_best_scale(cfg: CN, values: np.ndarray, scales: np.ndarray = np.arange(1, 9)):
    scales = np.sort(scales)
    scaled_values = np.expand_dims(values, axis=-1) * scales
    diff = np.abs(np.rint(scaled_values) - scaled_values)

    inlier_mask = diff < cfg.RANSAC.OFFSET_TOLERANCE / scales
    num_inliers = np.sum(inlier_mask, axis=tuple(range(inlier_mask.ndim - 1)))

    best_num_inliers = np.max(num_inliers)

    # We will choose a slightly worse scale if it is lower
    index = np.argmax(num_inliers > (
        1 - cfg.RANSAC.BEST_SOLUTION_TOLERANCE) * best_num_inliers)
    return scales[index], inlier_mask[..., index]


def discard_outliers(cfg: CN, warped_points: np.ndarray, intersection_points: np.ndarray) -> typing.Tuple[np.ndarray, np.ndarray, float, float]:
    horizontal_scale, horizontal_mask = find_best_scale(
        cfg, warped_points[..., 0])
    vertical_scale, vertical_mask = find_best_scale(cfg, warped_points[..., 1])
    mask = horizontal_mask & vertical_mask

    # Keep rows/cols that have more than 50% inliers
    num_rows_to_consider = np.any(mask, axis=-1).sum()
    num_cols_to_consider = np.any(mask, axis=-2).sum()
    rows_to_keep = mask.sum(axis=-1) / num_rows_to_consider > \
        cfg.MAX_OUTLIER_INTERSECTION_POINT_RATIO_PER_LINE
    cols_to_keep = mask.sum(axis=-2) / num_cols_to_consider > \
        cfg.MAX_OUTLIER_INTERSECTION_POINT_RATIO_PER_LINE

    warped_points = warped_points[rows_to_keep][:, cols_to_keep]
    intersection_points = intersection_points[rows_to_keep][:, cols_to_keep]
    return warped_points, intersection_points, horizontal_scale, vertical_scale


def quantize_points(warped_points: np.ndarray, horizontal_scale: float, vertical_scale: float) -> typing.Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean_col_xs = warped_points[..., 0].mean(axis=0)
    mean_row_ys = warped_points[..., 1].mean(axis=1)

    col_xs = np.rint(mean_col_xs * horizontal_scale)
    row_ys = np.rint(mean_row_ys * vertical_scale)

    quantized_points = np.stack(np.meshgrid(col_xs, row_ys), axis=-1)
    quantized_points[..., 0] = quantized_points[..., 0] / horizontal_scale
    quantized_points[..., 1] = quantized_points[..., 1] / vertical_scale

    return col_xs, row_ys, quantized_points


def _distance_from_point_to_line(p1: np.ndarray, p2: np.ndarray, point: np.ndarray):
    x0, y0 = np.moveaxis(point, -1, 0)
    x1, y1 = np.moveaxis(p1, -1, 0)
    x2, y2 = np.moveaxis(p2, -1, 0)
    numerator = np.abs((y2-y1)*x0 - (x2-x1)*y0 + x2*y1 - y2*x1)
    denominator = np.sqrt((y2-y1)**2 + (x2-x1)**2)
    return numerator / denominator


def get_edge_count_on_line(cfg: CN, edges: np.ndarray, scaled_p1: np.ndarray, scaled_p2: np.ndarray, horizontal_scale: float, vertical_scale: float, inverse_transformation_matrix: np.ndarray):
    line_points = np.stack([scaled_p1, scaled_p2], axis=0)
    line_points[..., 0] = line_points[..., 0] / horizontal_scale
    line_points[..., 1] = line_points[..., 1] / vertical_scale
    img_line_points = warp_points(inverse_transformation_matrix, line_points)
    x, y = np.meshgrid(np.arange(edges.shape[1]), np.arange(edges.shape[0]))
    points = np.stack([x, y], axis=-1)

    mask = _distance_from_point_to_line(
        *img_line_points, points) <= cfg.LINE_REFINEMENT.LINE_THRESHOLD
    return (edges & mask).sum()


def compute_vertical_borders(cfg: CN, col_xs: np.ndarray, row_ys: np.ndarray, edges: np.ndarray, horizontal_scale: float, vertical_scale: float, inverse_transformation_matrix: np.ndarray):
    xmin = col_xs.min()
    xmax = col_xs.max()
    ymin = row_ys.min()
    ymax = row_ys.max()

    def get_edge_count_on_vertical_line(x):
        points = np.array([[x, ymin],
                           [x, ymax]])
        return get_edge_count_on_line(cfg, edges, *points,
                                      horizontal_scale=horizontal_scale,
                                      vertical_scale=vertical_scale,
                                      inverse_transformation_matrix=inverse_transformation_matrix)

    while xmax - xmin < 8:
        left_points = get_edge_count_on_vertical_line(xmin - 1)
        right_points = get_edge_count_on_vertical_line(xmax + 1)
        if left_points > right_points:
            xmin -= 1
        else:
            xmax += 1
    return xmin, xmax


def compute_horizontal_borders(cfg: CN, col_xs: np.ndarray, row_ys: np.ndarray, edges: np.ndarray, horizontal_scale: float, vertical_scale: float, inverse_transformation_matrix: np.ndarray):
    xmin = col_xs.min()
    xmax = col_xs.max()
    ymin = row_ys.min()
    ymax = row_ys.max()

    def get_edge_count_on_horizontal_line(y):
        points = np.array([[xmin, y],
                           [xmax, y]])
        return get_edge_count_on_line(cfg, edges, *points,
                                      horizontal_scale=horizontal_scale,
                                      vertical_scale=vertical_scale,
                                      inverse_transformation_matrix=inverse_transformation_matrix)

    while ymax - ymin < 8:
        top_points = get_edge_count_on_horizontal_line(ymin - 1)
        bottom_points = get_edge_count_on_horizontal_line(ymax + 1)
        if top_points > bottom_points:
            ymin -= 1
        else:
            ymax += 1
    return ymin, ymax


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import argparse

    parser = argparse.ArgumentParser(description="Chessboard corner detector.")
    parser.add_argument("file", type=str, help="URI of the input image file")
    args = parser.parse_args()

    filename = URI(args.file)
    img = cv2.imread(str(filename))
    corners = find_corners(img)

    plt.figure()
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.scatter(*corners.T, c="r")
    plt.show()
