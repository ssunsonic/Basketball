import numpy as np
import cv2

def draw_shot_arc(frame, ball_positions):
    if len(ball_positions) < 5:
        return  # Not enough data to fit a curve

    # Extract the last 10 ball positions
    x_vals = [pos[0] for pos in list(ball_positions)[-10:]]
    y_vals = [pos[1] for pos in list(ball_positions)[-10:]]

    # Fit a quadratic curve (y = ax^2 + bx + c)
    coeffs = np.polyfit(x_vals, y_vals, 2)
    poly = np.poly1d(coeffs)

    # Predict future ball positions
    future_x = np.linspace(min(x_vals), max(x_vals) + 50, 15)  # Extend prediction
    future_y = poly(future_x)

    # Draw the trajectory curve
    for x, y in zip(future_x.astype(int), future_y.astype(int)):
        if 0 <= x < frame.shape[1] and 0 <= y < frame.shape[0]:  # Ensure points are within bounds
            cv2.circle(frame, (x, y), 3, (255, 0, 0), -1)  # Blue shot arc
