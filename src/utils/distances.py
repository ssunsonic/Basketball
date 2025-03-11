import math
import cv2
import os

def distance(p1, p2):
    """
    Args:
    p1 (x,y) and p2 (x,y)
    """
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def is_increasing_distances(point, points_array):
    """
    Args:
    point (tuple): (x,y)
    points_array (list): List of tuples [(x_1, y_1), (x_2, y_2), ...].

    Returns:
    bool: True if the distances are strictly increasing, False otherwise.
    """

    # Calculate distances between (x, y) and all points in the array
    distances = [distance(point, (x_i, y_i)) for (x_i, y_i) in points_array]

    # Check if distances are strictly increasing
    for i in range(1, len(distances)):
        if distances[i] <= distances[i - 1]:
            return False
    return True