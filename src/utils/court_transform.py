import cv2
import numpy as np

def detect_court_lines(frame):
    """Detects court lines using edge detection and Hough Transform."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)  # Reduce noise
    edges = cv2.Canny(blurred, 50, 150)  # Apply edge detection

    return edges  # Returns edge-detected frame

def detect_three_point_line(edges, frame):
    """Detects the three-point line using Hough Transform."""
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=100, maxLineGap=50)

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Draw detected lines

    return frame  # Returns frame with detected lines

