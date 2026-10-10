import math


def point_in_bbox(x, y, box):
    return box["x1"] <= x <= box["x2"] and box["y1"] <= y <= box["y2"]


def bbox_center(box):
    return (box["x1"] + box["x2"]) / 2, (box["y1"] + box["y2"]) / 2


def bbox_diagonal(box):
    width = box["x2"] - box["x1"]
    height = box["y2"] - box["y1"]
    return float((width * width + height * height) ** 0.5)


def point_to_bbox_center_distance(x, y, box):
    cx, cy = bbox_center(box)
    return float(((x - cx) ** 2 + (y - cy) ** 2) ** 0.5)


def gaussian_score(distance, sigma):
    if sigma <= 0:
        return 0.0
    return float(math.exp(-(distance**2) / (2 * sigma**2)))


def gaussian_overlap_density(distance, sigma_a, sigma_b):
    combined_variance = sigma_a**2 + sigma_b**2
    if combined_variance <= 0:
        return 0.0
    normalizer = 1.0 / (2.0 * math.pi * combined_variance)
    return float(normalizer * math.exp(-(distance**2) / (2.0 * combined_variance)))
