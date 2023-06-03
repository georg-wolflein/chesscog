"""Module to perform chessboard localization.

The entire board localization pipeline is implemented end-to-end in this module.
Use the :meth:`find_corners` to run it.

This module simultaneously acts as a script:

.. code-block:: console

    $ python -m chesscog.corner_detection.detect_corners --help
    usage: detect_corners.py [-h] [--config CONFIG] file
    
    Chessboard corner detector.
    
    positional arguments:
      file             URI of the input image file
    
    optional arguments:
      -h, --help       show this help message and exit
      --config CONFIG  path to the config file
"""

from sklearn.metrics.pairwise import pairwise_distances
from sklearn.cluster import AgglomerativeClustering, DBSCAN
import cv2
import numpy as np
import typing
from recap import URI, CfgNode as CN

from chesscog.core import sort_corner_points
from chesscog.core.coordinates import from_homogenous_coordinates, to_homogenous_coordinates
from chesscog.core.exceptions import ChessboardNotLocatedException


def find_corners(cfg: CN, img: np.ndarray) -> np.ndarray:
    """Determine the four corner points of the chessboard in an image.

    Args:
        cfg (CN): the configuration object
        img (np.ndarray): the input image (as a numpy array)

    Raises:
        ChessboardNotLocatedException: if the chessboard could not be found

    Returns:
        np.ndarray: the pixel coordinates of the four corners
    """
    img, img_scale = resize_image(cfg, img)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = _detect_edges(cfg.EDGE_DETECTION, gray)
    lines = _detect_lines(cfg, edges)
    if lines.shape[0] > 400:
        raise ChessboardNotLocatedException("too many lines in the image")
    all_horizontal_lines, all_vertical_lines = _cluster_horizontal_and_vertical_lines(
        lines)

    horizontal_lines = _eliminate_similar_lines(
        all_horizontal_lines, all_vertical_lines)
    vertical_lines = _eliminate_similar_lines(
        all_vertical_lines, all_horizontal_lines)

    all_intersection_points = _get_intersection_points(horizontal_lines,
                                                       vertical_lines)

    best_num_inliers = 0
    best_configuration = None
    iterations = 0
    while iterations < 200 or best_num_inliers < 30:
        row1, row2 = _choose_from_range(len(horizontal_lines))
        col1, col2 = _choose_from_range(len(vertical_lines))
        transformation_matrix = _compute_homography(all_intersection_points,
                                                    row1, row2, col1, col2)
        warped_points = _warp_points(
            transformation_matrix, all_intersection_points)
        warped_points, intersection_points, horizontal_scale, vertical_scale = _discard_outliers(
            cfg, warped_points, all_intersection_points)
        num_inliers = np.prod(warped_points.shape[:-1])
        if num_inliers > best_num_inliers:
            warped_points *= np.array((horizontal_scale, vertical_scale))

            # Quantize and reject deuplicates
            (xmin, xmax, ymin, ymax), scale, quantized_points, intersection_points, warped_img_size = configuration = _quantize_points(
                cfg, warped_points, intersection_points)

            # Calculate remaining number of inliers
            num_inliers = np.prod(quantized_points.shape[:-1])

            if num_inliers > best_num_inliers:
                best_num_inliers = num_inliers
                best_configuration = configuration
        iterations += 1
        if iterations > 10000:
            raise ChessboardNotLocatedException(
                "RANSAC produced no viable results")

    # Retrieve best configuration
    (xmin, xmax, ymin, ymax), scale, quantized_points, intersection_points, warped_img_size = best_configuration

    # Recompute transformation matrix based on all inliers
    transformation_matrix = compute_transformation_matrix(
        intersection_points, quantized_points)
    inverse_transformation_matrix = np.linalg.inv(transformation_matrix)

    # Warp grayscale image
    dims = tuple(warped_img_size.astype(np.int32))
    warped = cv2.warpPerspective(gray, transformation_matrix, dims)
    borders = np.zeros_like(gray)
    borders[3:-3, 3:-3] = 1
    warped_borders = cv2.warpPerspective(borders, transformation_matrix, dims)
    warped_mask = warped_borders == 1

    # Refine board boundaries
    xmin, xmax = _compute_vertical_borders(
        cfg, warped, warped_mask, scale, xmin, xmax)
    scaled_xmin, scaled_xmax = (int(x * scale[0]) for x in (xmin, xmax))
    warped_mask[:, :scaled_xmin] = warped_mask[:, scaled_xmax:] = False
    ymin, ymax = _compute_horizontal_borders(
        cfg, warped, warped_mask, scale, ymin, ymax)

    # Transform boundaries to image space
    corners = np.array([[xmin, ymin],
                        [xmax, ymin],
                        [xmax, ymax],
                        [xmin, ymax]]).astype(np.float32)
    corners = corners * scale
    img_corners = _warp_points(inverse_transformation_matrix, corners)
    img_corners = img_corners / img_scale
    return sort_corner_points(img_corners)


def resize_image(cfg: CN, img: np.ndarray) -> typing.Tuple[np.ndarray, float]:
    """Resize an image for use in the corner detection pipeline, maintaining the aspect ratio.

    Args:
        cfg (CN): the configuration object
        img (np.ndarray): the input image

    Returns:
        typing.Tuple[np.ndarray, float]: the resized image along with the scale of this new image
    """
    h, w, _ = img.shape
    if w == cfg.RESIZE_IMAGE.WIDTH:
        return img, 1
    scale = cfg.RESIZE_IMAGE.WIDTH / w
    dims = (cfg.RESIZE_IMAGE.WIDTH, int(h * scale))

    img = cv2.resize(img, dims)
    return img, scale


def _detect_edges(edge_detection_cfg: CN, gray: np.ndarray) -> np.ndarray:
    if gray.dtype != np.uint8:
        gray = gray / gray.max() * 255
        gray = gray.astype(np.uint8)
    edges = cv2.Canny(gray,
                      edge_detection_cfg.LOW_THRESHOLD,
                      edge_detection_cfg.HIGH_THRESHOLD,
                      edge_detection_cfg.APERTURE)
    return edges


def _detect_lines(cfg: CN, edges: np.ndarray) -> np.ndarray:
    # array of [rho, theta]
    lines = cv2.HoughLines(edges, 1, np.pi/360, cfg.LINE_DETECTION.THRESHOLD)
    lines = lines.squeeze(axis=-2)
    lines = _fix_negative_rho_in_hesse_normal_form(lines)

    if cfg.LINE_DETECTION.DIAGONAL_LINE_ELIMINATION:
        threshold = np.deg2rad(
            cfg.LINE_DETECTION.DIAGONAL_LINE_ELIMINATION_THRESHOLD_DEGREES)
        vmask = np.abs(lines[:, 1]) < threshold
        hmask = np.abs(lines[:, 1] - np.pi / 2) < threshold
        mask = vmask | hmask
        lines = lines[mask]
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


def _sort_lines(lines: np.ndarray) -> np.ndarray:
    if lines.ndim == 0 or lines.shape[-2] == 0:
        return lines
    rhos = lines[..., 0]
    sorted_indices = np.argsort(rhos)
    return lines[sorted_indices]


def _cluster_horizontal_and_vertical_lines(lines: np.ndarray):
    lines = _sort_lines(lines)
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

    return horizontal_lines, vertical_lines


def _eliminate_similar_lines(lines: np.ndarray, perpendicular_lines: np.ndarray) -> np.ndarray:
    perp_rho, perp_theta = perpendicular_lines.mean(axis=0)
    rho, theta = np.moveaxis(lines, -1, 0)
    intersection_points = get_intersection_point(
        rho, theta, perp_rho, perp_theta)
    intersection_points = np.stack(intersection_points, axis=-1)

    clustering = DBSCAN(eps=12, min_samples=1).fit(intersection_points)

    filtered_lines = []
    for c in range(clustering.labels_.max() + 1):
        lines_in_cluster = lines[clustering.labels_ == c]
        rho = lines_in_cluster[..., 0]
        median = np.argsort(rho)[len(rho)//2]
        filtered_lines.append(lines_in_cluster[median])
    return np.stack(filtered_lines)


def get_intersection_point(rho1: np.ndarray, theta1: np.ndarray, rho2: np.ndarray, theta2: np.ndarray) -> typing.Tuple[np.ndarray, np.ndarray]:
    """Obtain the intersection point of two lines in Hough space.

    This method can be batched

    Args:
        rho1 (np.ndarray): first line's rho
        theta1 (np.ndarray): first line's theta
        rho2 (np.ndarray): second lines's rho
        theta2 (np.ndarray): second line's theta

    Returns:
        typing.Tuple[np.ndarray, np.ndarray]: the x and y coordinates of the intersection point(s)
    """
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


def _get_intersection_points(horizontal_lines: np.ndarray, vertical_lines: np.ndarray) -> np.ndarray:
    rho1, theta1 = np.moveaxis(horizontal_lines, -1, 0)
    rho2, theta2 = np.moveaxis(vertical_lines, -1, 0)

    rho1, rho2 = np.meshgrid(rho1, rho2, indexing="ij")
    theta1, theta2 = np.meshgrid(theta1, theta2, indexing="ij")
    intersection_points = get_intersection_point(rho1, theta1, rho2, theta2)
    intersection_points = np.stack(intersection_points, axis=-1)
    return intersection_points


def compute_transformation_matrix(src_points: np.ndarray, dst_points: np.ndarray) -> np.ndarray:
    """Compute the transformation matrix based on source and destination points.

    Args:
        src_points (np.ndarray): the source points (shape: [..., 2])
        dst_points (np.ndarray): the source points (shape: [..., 2])

    Returns:
        np.ndarray: the transformation matrix
    """
    transformation_matrix, _ = cv2.findHomography(src_points.reshape(-1, 2),
                                                  dst_points.reshape(-1, 2))
    return transformation_matrix


def _compute_homography(intersection_points: np.ndarray, row1: int, row2: int, col1: int, col2: int):
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


def _warp_points(transformation_matrix: np.ndarray, points: np.ndarray) -> np.ndarray:
    points = to_homogenous_coordinates(points)
    warped_points = points @ transformation_matrix.T
    return from_homogenous_coordinates(warped_points)


def _find_best_scale(cfg: CN, values: np.ndarray, scales: np.ndarray = np.arange(1, 9)):
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


def _discard_outliers(cfg: CN, warped_points: np.ndarray, intersection_points: np.ndarray) -> typing.Tuple[np.ndarray, np.ndarray, float, float]:
    horizontal_scale, horizontal_mask = _find_best_scale(
        cfg, warped_points[..., 0])
    vertical_scale, vertical_mask = _find_best_scale(
        cfg, warped_points[..., 1])
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


def _quantize_points(cfg: CN, warped_scaled_points: np.ndarray, intersection_points: np.ndarray) -> typing.Tuple[tuple, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mean_col_xs = warped_scaled_points[..., 0].mean(axis=0)
    mean_row_ys = warped_scaled_points[..., 1].mean(axis=1)

    col_xs = np.rint(mean_col_xs).astype(np.int32)
    row_ys = np.rint(mean_row_ys).astype(np.int32)

    # Remove duplicates
    col_xs, col_indices = np.unique(col_xs, return_index=True)
    row_ys, row_indices = np.unique(row_ys, return_index=True)
    intersection_points = intersection_points[row_indices][:, col_indices]

    # Compute mins and maxs in warped space
    xmin = col_xs.min()
    xmax = col_xs.max()
    ymin = row_ys.min()
    ymax = row_ys.max()

    # Ensure we a have a maximum of 9 rows/cols
    while xmax - xmin > 9:
        xmax -= 1
        xmin += 1
    while ymax - ymin > 9:
        ymax -= 1
        ymin += 1
    col_mask = (col_xs >= xmin) & (col_xs <= xmax)
    row_mask = (row_ys >= xmin) & (row_ys <= xmax)

    # Discard
    col_xs = col_xs[col_mask]
    row_ys = row_ys[row_mask]
    intersection_points = intersection_points[row_mask][:, col_mask]

    # Create quantized points array
    quantized_points = np.stack(np.meshgrid(col_xs, row_ys), axis=-1)

    # Transform in warped space
    translation = -np.array([xmin, ymin]) + \
        cfg.BORDER_REFINEMENT.NUM_SURROUNDING_SQUARES_IN_WARPED_IMG
    scale = np.array(cfg.BORDER_REFINEMENT.WARPED_SQUARE_SIZE)

    scaled_quantized_points = (quantized_points + translation) * scale
    xmin, ymin = np.array((xmin, ymin)) + translation
    xmax, ymax = np.array((xmax, ymax)) + translation
    warped_img_size = (np.array((xmax, ymax)) +
                       cfg.BORDER_REFINEMENT.NUM_SURROUNDING_SQUARES_IN_WARPED_IMG) * scale

    return (xmin, xmax, ymin, ymax), scale, scaled_quantized_points, intersection_points, warped_img_size


def _compute_vertical_borders(cfg: CN, warped: np.ndarray, mask: np.ndarray, scale: np.ndarray, xmin: int, xmax: int) -> typing.Tuple[int, int]:
    G_x = np.abs(cv2.Sobel(warped, cv2.CV_64F, 1, 0,
                           ksize=cfg.BORDER_REFINEMENT.SOBEL_KERNEL_SIZE))
    G_x[~mask] = 0
    G_x = _detect_edges(cfg.BORDER_REFINEMENT.EDGE_DETECTION.VERTICAL, G_x)
    G_x[~mask] = 0

    def get_nonmax_supressed(x):
        x = (x * scale[0]).astype(np.int32)
        thresh = cfg.BORDER_REFINEMENT.LINE_WIDTH // 2
        return G_x[:, x-thresh:x+thresh+1].max(axis=1)

    while xmax - xmin < 8:
        top = get_nonmax_supressed(xmax + 1)
        bottom = get_nonmax_supressed(xmin - 1)

        if top.sum() > bottom.sum():
            xmax += 1
        else:
            xmin -= 1

    return xmin, xmax


def _compute_horizontal_borders(cfg: CN, warped: np.ndarray, mask: np.ndarray, scale: np.ndarray, ymin: int, ymax: int) -> typing.Tuple[int, int]:
    G_y = np.abs(cv2.Sobel(warped, cv2.CV_64F, 0, 1,
                           ksize=cfg.BORDER_REFINEMENT.SOBEL_KERNEL_SIZE))
    G_y[~mask] = 0
    G_y = _detect_edges(cfg.BORDER_REFINEMENT.EDGE_DETECTION.HORIZONTAL, G_y)
    G_y[~mask] = 0

    def get_nonmax_supressed(y):
        y = (y * scale[1]).astype(np.int32)
        thresh = cfg.BORDER_REFINEMENT.LINE_WIDTH // 2
        return G_y[y-thresh:y+thresh+1].max(axis=0)

    while ymax - ymin < 8:
        top = get_nonmax_supressed(ymax + 1)
        bottom = get_nonmax_supressed(ymin - 1)

        if top.sum() > bottom.sum():
            ymax += 1
        else:
            ymin -= 1
    return ymin, ymax


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import argparse

    parser = argparse.ArgumentParser(description="Chessboard corner detector.")
    parser.add_argument("file", type=str, help="URI of the input image file")
    parser.add_argument("--config", type=str, help="path to the config file",
                        default="config://corner_detection.yaml")
    args = parser.parse_args()

    cfg = CN.load_yaml_with_base(args.config)
    filename = URI(args.file)
    img = cv2.imread(str(filename))
    corners = find_corners(cfg, img)

    fig = plt.figure()
    fig.canvas.set_window_title("Corner detection output")
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.scatter(*corners.T, c="r")
    plt.axis("off")
    plt.show()
