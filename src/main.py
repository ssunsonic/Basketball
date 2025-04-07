import math
import random
import os
from collections import deque
from typing import Tuple, Optional

import cv2
import numpy as np
from ultralytics import YOLO
from sort.sort import Sort
import ipdb
from audioplayer import AudioPlayer

from src.utils.distances import distance


class BasketballTracker:
    """
    Basketball shot detection and tracking system.

    This class handles tracking of players, shots, and shot outcomes
    in basketball games.
    """

    def __init__(
        self,
        model_path: str = "models/bball.pt",
        video_path: str = None,
        camera_id: int = None,
        output_dir: str = "output_vids",
        frame_make_cooldown: int = 60,
        frame_shoot_cooldown: int = 60,
        multiple: bool = False,  # Add multiple parameter
        rim_overlap_threshold: float = 95.0,  # Percentage threshold for rim overlap detection
    ):
        """
        Initialize the Basketball Tracker.

        Args:
            model_path: Path to the YOLO model
            video_path: Path to the input video (None for camera)
            camera_id: Camera ID for live tracking (None for video file)
            output_dir: Directory to save output videos
            frame_make_cooldown: Frames to wait between made shots
            frame_shoot_cooldown: Frames to wait between shot attempts
            multiple: Whether to use multiple player UI mode
            rim_overlap_threshold: Percentage of ball area that must overlap with rim for shot detection
        """
        self.model = YOLO(model_path)
        self.labels = self.model.names
        self.tracker = Sort(max_age=60, min_hits=1, iou_threshold=0.3)
        self.multiple = multiple  # Flag for multiple player UI mode
        self.rim_overlap_threshold = (
            rim_overlap_threshold  # Threshold for rim overlap detection
        )
        self.last_overlap_percentage = (
            0.0  # Track the last calculated overlap percentage
        )

        # Initialize video capture
        if video_path:
            self.cap = cv2.VideoCapture(video_path)
        elif camera_id is not None:
            self.cap = cv2.VideoCapture(camera_id)
        else:
            raise ValueError("Either video_path or camera_id must be provided")

        # Video properties
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))

        # Output settings
        self.output_dir = output_dir
        self.output_path = self.get_available_filename(output_dir, "output", "mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.out = cv2.VideoWriter(
            self.output_path, fourcc, self.fps, (self.frame_width, self.frame_height)
        )
        if not self.out.isOpened():
            raise RuntimeError("Error: Could not open video writer.")

        # Tracking core metrics
        self.shot_counter = 0
        self.shots_made = 0
        self.shot_in_progress = False

        # SORT tracking
        self.tracked_objects = np.empty((0, 5))

        # Frame counters
        self.frame_counter = 0
        # Frame Cooldowns
        self.frame_since_last_make = 0
        self.frame_since_last_shot = 0
        self.frame_make_cooldown = frame_make_cooldown
        self.frame_shoot_cooldown = frame_shoot_cooldown

        # Player tracking
        self.players = {}  # {id: (x, y)} player positions
        self.player_shots = {}  # {id: shots_count} shots per player
        self.player_makes = {}  # {id: makes_count} makes per player
        self.unique_ids = set()  # Unique player IDs
        self.current_shooter_id = None

        # Position tracking
        self.ball_position = deque(maxlen=30)
        self.rim_position = deque(maxlen=30)
        self.shot_position = deque(maxlen=30)
        self.post_shot_ball_positions = deque(maxlen=30) # For drawing trajectory mask

        # Shot detection enhancement
        self.shot_in_progress_since = 0  # Frame when shot started
        self.ball_near_rim = False  # Flag for when ball is near rim
        self.ball_below_rim_after_shot = (
            False  # Flag for checking if ball passed through rim
        )
        self.ball_tracking_buffer = []  # Track ball positions after shot
        self.max_tracking_frames = 500  # Track ball for this many frames after shot
        self.last_shot_frame = 0  # Frame when the last shot was detected

        # Visualization
        self.ball_mask = None
        self.shot_frames = 0
        self.shot_made_indicator = None

    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, bool, bool]:
        """
        Process a single frame of video.

        Args:
            frame: Input video frame

        Returns:
            Tuple containing:
                - Processed frame
                - Whether a shot was attempted
                - Whether a shot was made
        """
        # Increment frame counters
        self.frame_since_last_make += 1
        self.frame_since_last_shot += 1
        self.frame_counter += 1

        # Initialize flags
        shot_made = False
        shot_attempt = False
        shooter_id = None

        # YOLO inference
        results = self.model(frame, stream=False)

        # Process results
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # Get coordinates
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                w, h = x2 - x1, y2 - y1  # Width and height of bounding box
                detections_sort = box.xyxy[0].numpy()

                # Get confidence score
                conf = math.ceil(box.conf[0] * 100) / 100
                conf_sort = box.conf.numpy()

                # Class info
                cls = int(box.cls[0])
                current_class = self.labels.get(cls)
                # cls_sort = box.cls.numpy()

                # Get centered coordinates
                cx, cy = x1 + w // 2, y1 + h // 2

                # Handle different detected classes
                if current_class == "shoot" and conf > 0.85:
                    shot_position = [cx, cy, self.frame_counter]
                    self.shot_position.append(shot_position)
                    shot_attempt = True
                    self.shot_in_progress = True
                    self.shot_frames = 60  # Duration of shot in frames
                    self.post_shot_ball_positions.clear()
                    self.ball_tracking_buffer = []  # Clear the buffer for new shot
                    self.ball_near_rim = False
                    self.ball_below_rim_after_shot = False
                    self.shot_in_progress_since = self.frame_counter
                    self.last_shot_frame = self.frame_counter

                    # Find closest player to shot position (shooter)
                    shooter_id = self._find_closest_player(cx, cy)

                    # Register shot if cooldown has passed
                    if self.frame_since_last_shot > self.frame_shoot_cooldown:
                        # Check that player is shooting with ball in hand
                        if len(self.ball_position) > 0 and len(self.shot_position) > 0:
                            ball_shot_distance = distance(
                                (self.shot_position[-1][0], self.shot_position[-1][1]),
                                (self.ball_position[-1][0], self.ball_position[-1][1]),
                            )
                            print(
                                f"Distance between ball and shot position: {ball_shot_distance}"
                            )
                            if ball_shot_distance < 500:
                                if shooter_id is not None:
                                    self.player_shots[shooter_id] = (
                                        self.player_shots.get(shooter_id, 0) + 1
                                    )
                                    print(f"🏀 Player {shooter_id} took a shot!")
                                self.frame_since_last_shot = 0
                                self.shot_counter += 1
                                self.current_shooter_id = shooter_id
                            else:
                                print(
                                    f"❌ False positive shot detection - ball too far from shooter: {ball_shot_distance} pixels"
                                )

                # Model-based detection - keep this as a fallback
                if current_class == "made":
                    print(
                        f"✅ Shot made by player {self.current_shooter_id}! at frame {self.frame_counter}"
                    )
                    shot_made = True
                    self.shot_in_progress = False

                if current_class == "rim" and conf > 0.70:
                    y1 += 20
                    self.rim_position.append([x1, x2, y1, y2, self.frame_counter])

                # Ball detection
                # "Made" logic didn't work because ball class was not detected in some cases

                if current_class == "ball" and conf > 0.3:
                    self.ball_position.append(
                        [cx, cy, self.frame_counter, w, h]
                    )  # Also store ball size
                    self.post_shot_ball_positions.append(
                        [cx, cy, self.frame_counter, w, h]
                    )

                    # Track ball for shot detection
                    if (
                        self.shot_in_progress
                        and self.frame_counter - self.shot_in_progress_since
                        < self.max_tracking_frames
                    ):
                        # Store ball position for trajectory analysis
                        self.ball_tracking_buffer.append(
                            (cx, cy, self.frame_counter, w, h)
                        )

                        # Check if we have a rim position to reference
                        if len(self.rim_position) > 0:
                            latest_rim = self.rim_position[-1]
                            rim_x = (
                                latest_rim[0] + latest_rim[1]
                            ) // 2  # Center x of rim
                            rim_y = latest_rim[2]  # Top y of rim
                            rim_bottom = latest_rim[3]  # Bottom y of rim
                            rim_radius = (latest_rim[1] - latest_rim[0]) // 2  # Same as the one used for visualization

                            # Calculate distance from ball to rim
                            dist_to_rim = math.sqrt(
                                (cx - rim_x) ** 2 + (cy - rim_y) ** 2
                            )

                            # DEBUGGING
                            # if self.frame_counter == 105:
                            #         ipdb.set_trace()

                            # Ball is near rim (within radius of the rim)
                            if (
                                dist_to_rim < 150
                            ):  # Adjust threshold based on your video scale
                                self.ball_near_rim = True
                                print(f"Ball near rim at frame {self.frame_counter}")
                                print(f"Distance to rim: {dist_to_rim}")
                                print(f"Ball vs Rim Y: ({cy}, {rim_y})")

                                # Calculate overlap percentage between ball and rim target area
                                ball_radius = min(w, h) / 2
                                ball_area = (
                                    math.pi * ball_radius**2
                                )  # Use smaller dimension for conservative estimate

                                # Calculate the distance between centers
                                center_distance = dist_to_rim

                                # If centers are close enough for potential overlap
                                if center_distance < (rim_radius + ball_radius):
                                    # Calculate intersection area of two circles
                                    # Formula: https://mathworld.wolfram.com/Circle-CircleIntersection.html
                                    d = center_distance
                                    r1 = rim_radius
                                    r2 = ball_radius

                                    if d <= abs(
                                        r1 - r2
                                    ):  # One circle is inside the other
                                        # Smaller circle is completely inside larger circle
                                        overlap_area = math.pi * min(r1, r2) ** 2
                                    elif d >= r1 + r2:  # Circles are separate
                                        overlap_area = 0
                                    else:
                                        # Partial overlap - calculate intersection area
                                        d1 = (r1**2 - r2**2 + d**2) / (2 * d)
                                        d2 = d - d1
                                        overlap_area = (
                                            r1**2 * math.acos(d1 / r1)
                                            - d1 * math.sqrt(r1**2 - d1**2)
                                            + r2**2 * math.acos(d2 / r2)
                                            - d2 * math.sqrt(r2**2 - d2**2)
                                        )

                                    # Calculate percentage of ball inside rim area
                                    if ball_area > 0:  # Prevent division by zero
                                        overlap_percentage = min(
                                            100, (overlap_area / ball_area) * 100
                                        )
                                        self.last_overlap_percentage = overlap_percentage  # Store for visualization
                                        print(
                                            f"Ball-rim overlap: {overlap_percentage:.2f}%"
                                        )

                                        # If overlap is significant (exceeds threshold), consider it a made shot
                                        if (
                                            overlap_percentage
                                            > self.rim_overlap_threshold
                                        ):
                                            self.ball_below_rim_after_shot = True
                                            print(
                                                f"Ball passed through rim at frame {self.frame_counter}"
                                            )
                                            print(
                                                f"Overlap percentage: {overlap_percentage:.2f}% (threshold: {self.rim_overlap_threshold:.2f}%)"
                                            )

                # SORT Detection for player tracking
                if current_class == "person":
                    detections_for_sort = np.array([[*detections_sort, conf_sort[0]]])
                    self.tracked_objects = self.tracker.update(detections_for_sort)

                    # Update player positions
                    self._update_player_positions(self.tracked_objects)

        # Check for trajectory-based made shot detection
        trajectory_shot_made = False
        if (
            self.shot_in_progress
            and self.ball_near_rim
            and self.ball_below_rim_after_shot
            and self.frame_counter - self.shot_in_progress_since
            >= 10  # Wait enough frames for ball to complete trajectory
            and self.frame_counter - self.shot_in_progress_since
            < self.max_tracking_frames
        ):  # Within tracking window

            # Trajectory analysis indicates a made shot
            trajectory_shot_made = True
            shot_made = True
            print(
                f"🏀 Trajectory analysis detected a made shot at frame {self.frame_counter}!"
            )

            # Reset flags after detection
            self.ball_below_rim_after_shot = False
            self.ball_near_rim = False

        # Handle shot made registration (from either model or trajectory detection)
        if (
            shot_made or trajectory_shot_made
        ) and self.frame_since_last_make > self.frame_make_cooldown:
            if self.current_shooter_id is not None:
                self.player_makes[self.current_shooter_id] = (
                    self.player_makes.get(self.current_shooter_id, 0) + 1
                )
                print(f"🎯 Player {self.current_shooter_id} made a shot!")

                # Lebron Lebron Lebron
                AudioPlayer("sounds/lebron.mp3").play(block=True)

                # Store shot made indicator details for displaying
                self.shot_made_indicator = {
                    "text": "+1",
                    "position": (
                        self.rim_position[-1][0] + random.randint(100, 125),
                        self.rim_position[-1][2] - random.randint(50, 100),
                    ),
                    "frames_left": 15,
                    "color": (0, 255, 0),
                }

            self.shots_made += 1
            self.frame_since_last_make = 0
            # Reset shot flags after registering
            self.shot_in_progress = False

            # Visualize the overlap calculation results when appropriate
            # if hasattr(self, "last_overlap_percentage"):
            #     cv2.putText(
            #         frame,
            #         f"Overlap: {self.last_overlap_percentage:.1f}%",
            #         (10, 240),
            #         cv2.FONT_HERSHEY_SIMPLEX,
            #         0.7,
            #         (255, 255, 0),
            #         2,
            #     )

        # Update ball trajectory visualization
        self._update_ball_trajectory(frame)

        # Draw player information and scores - Pass the multiple parameter here
        processed_frame = self._draw_ui_elements(frame, self.multiple)

        return processed_frame, shot_attempt, shot_made

    def _find_closest_player(self, x: int, y: int) -> Optional[int]:
        """
        Find the closest player to a given position.

        Args:
            x: X-coordinate
            y: Y-coordinate

        Returns:
            Player ID of closest player or None if no players found
        """
        min_distance = float("inf")
        closest_player = None

        for player_id, (px, py) in self.players.items():
            distance = ((px - x) ** 2 + (py - y) ** 2) ** 0.5  # Euclidean distance
            if distance < min_distance:
                min_distance = distance
                closest_player = player_id

        return closest_player

    def _update_player_positions(self, tracked_objects: np.ndarray) -> None:
        """
        Update player positions based on tracking results.

        Args:
            tracked_objects: Array of tracked object coordinates and IDs
        """
        for x1, y1, x2, y2, obj_id in tracked_objects:
            obj_id = int(obj_id)
            w = int(x2) - int(x1)
            cx, cy = int(x1) + w // 2, int(y1)  # Get player center position
            self.players[obj_id] = (cx, cy)  # Store player position
            self.unique_ids.add(obj_id)

    def _update_ball_trajectory(self, frame: np.ndarray) -> None:
        """
        Update the ball trajectory visualization.

        Args:
            frame: Current video frame
        """
        if self.ball_mask is None:
            self.ball_mask = np.zeros_like(frame, dtype=np.uint8)

        if self.shot_in_progress:
            if self.frame_counter % 5 == 0:
                # Clear the overlay (reset to transparent)
                self.ball_mask = np.zeros_like(frame, dtype=np.uint8)

                # Draw trajectory points
                for pos in self.post_shot_ball_positions:
                    cx, cy, pos_frame, w, h = pos
                    if pos_frame % 5 == 0:
                        cv2.circle(
                            self.ball_mask,
                            (int(cx), int(cy)),
                            5,
                            (0, 0, 255),
                            cv2.FILLED,
                        )

                # Draw rim if available (for debugging)
                if len(self.rim_position) > 0:
                    latest_rim = self.rim_position[-1]
                    rim_x1 = int(latest_rim[0])
                    rim_x2 = int(latest_rim[1])
                    rim_y = int(latest_rim[2])
                    rim_bottom = int(latest_rim[3])
                    rim_center_x = (rim_x1 + rim_x2) // 2

                    # Draw rim rectangle
                    cv2.rectangle(
                        self.ball_mask,
                        (rim_x1, rim_y),
                        (rim_x2, rim_bottom),
                        (0, 255, 0),
                        2,
                    )

                    # Draw detection circles for rim area
                    rim_radius = (rim_x2 - rim_x1) // 2
                    cv2.circle(
                        self.ball_mask,
                        (rim_center_x, rim_y),
                        rim_radius,
                        (0, 255, 255),
                        1,
                    )  # Rim detection circle

            # Decrease shot frames
            self.shot_frames -= 1
            if self.shot_frames == 0:
                # self.shot_in_progress = False
                self.post_shot_ball_positions.clear()
                self.ball_mask = np.zeros_like(frame, dtype=np.uint8)

    def _draw_ui_elements(
        self, frame: np.ndarray, multiple: bool = False
    ) -> np.ndarray:
        """
        Draw UI elements on the frame including player info and scores.

        Args:
            frame: Current video frame
            multiple: Whether to use multiple player UI mode

        Returns:
            Frame with UI elements added
        """
        # Create a copy of the frame
        display_frame = frame.copy()

        # Draw shot made indicator if needed
        if self.shot_made_indicator and self.shot_made_indicator["frames_left"] > 0:
            cv2.putText(
                display_frame,
                self.shot_made_indicator["text"],
                self.shot_made_indicator["position"],
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                self.shot_made_indicator["color"],
                2,
            )

            # Decrease frames counter
            self.shot_made_indicator["frames_left"] -= 1

            # Change color randomly for visual effect
            if self.shot_made_indicator["frames_left"] > 0:
                colors = [
                    (0, 255, 0), # Green
                    (255, 0, 0), # Blue
                    (0, 0, 255), # Red
                    (0, 255, 255), # Yellow
                    (255, 0, 255) # Magenta
                ]
                self.shot_made_indicator["color"] = random.choice(colors)

        if multiple:
            # For multiple players mode, don't draw the stats background or general shot statistics
            # This clears the left side box and text
            pass
        else:
            # Draw stats background
            cv2.rectangle(display_frame, (5, 0), (400, 150), (255, 255, 255), -1)

            # Draw shot statistics
            cv2.putText(
                display_frame,
                f"Shot Attempted: {self.shot_counter}",
                (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 0),
                3,
            )
            cv2.putText(
                display_frame,
                f"Shots Made: {self.shots_made}",
                (10, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 0),
                3,
            )

            # Shooting percentage
            if self.shot_counter > 0:
                shooting_pct = self.shots_made / self.shot_counter
                cv2.putText(
                    display_frame,
                    f"Shooting PCT: {100 * shooting_pct:.2f}%",
                    (10, 120),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 0),
                    3,
                )
            else:
                cv2.putText(
                    display_frame,
                    "Shooting PCT: 0%",
                    (10, 120),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 0),
                    3,
                )

        # Multiple Players
        cv2.rectangle(display_frame, (3300, 0), (4000, 250), (255, 255, 255), -1)

        # Draw tracked players with bounding boxes and IDs
        for obj_id, (cx, cy) in self.players.items():
            # Find corresponding tracking box
            for x1, y1, x2, y2, track_id in self.tracked_objects:
                if int(track_id) == obj_id and int(track_id) in [1, 2]:
                    # Calculate width and height
                    w = int(x2) - int(x1)
                    # h = int(y2) - int(y1)

                    # Player color - different color for each player
                    color = (255, 0, 0) if obj_id == 1 else (0, 0, 255)

                    # Draw player bounding box
                    cv2.rectangle(
                        display_frame,
                        (int(x1), int(y1)),
                        (int(x2), int(y2)),
                        color,
                        2,
                    )

                    # Draw player ID
                    cv2.putText(
                        display_frame,
                        f"Player {obj_id}",
                        (int(x1) + w // 4, int(y1) - 20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        color,
                        3,
                    )
                    break

        # Draw individual player scores in top right
        y_offset = 100

        # Visualize player statistics
        for player_id in sorted(self.unique_ids):
            if player_id in [1, 2]:
                shots = self.player_shots.get(player_id, 0)
                makes = self.player_makes.get(player_id, 0)
                color = (255, 0, 0) if player_id == 1 else (0, 0, 255)

                cv2.putText(
                    display_frame,
                    f"Player {player_id}: {makes}/{shots}",
                    (self.frame_width - 500, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    2,
                    color,
                    3,
                )
                y_offset += 100

        # Blend the ball trajectory mask onto the frame
        if self.ball_mask is not None:
            display_frame = cv2.addWeighted(display_frame, 1, self.ball_mask, 0.75, 0.5)

        return display_frame

    def get_available_filename(
        self, directory: str, base_name: str, extension: str
    ) -> str:
        """
        Generate a unique filename that doesn't exist yet.

        Args:
            directory: Output directory path
            base_name: Base filename
            extension: File extension

        Returns:
            str: Available file path
        """
        os.makedirs(directory, exist_ok=True)

        counter = 0
        while True:
            if counter == 0:
                filename = f"{base_name}.{extension}"
            else:
                filename = f"{base_name}_{counter}.{extension}"

            filepath = os.path.join(directory, filename)
            if not os.path.exists(filepath):
                return filepath
            counter += 1

    def run(self, rotate_video: bool = False) -> None:
        """
        Run the basketball tracking system on the video/camera input.

        Args:
            rotate_video: Whether to rotate the video 90 degrees clockwise
        """
        try:
            while self.cap.isOpened():
                ret, frame = self.cap.read()
                if not ret:
                    break

                # For portrait videos
                if rotate_video:
                    frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

                # Process the current frame
                processed_frame, _, _ = self.process_frame(frame)

                # Display the frame
                cv2.imshow("Basketball Shot Tracking", processed_frame)

                # Write to output video
                self.out.write(processed_frame)

                # Exit on 'q' press
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

        except KeyboardInterrupt:
            print("\n[INFO] Ctrl+C detected. Saving video and closing...")

        finally:
            self.cleanup()

    def cleanup(self) -> None:
        """Release resources and print summary statistics."""
        # Release resources
        self.cap.release()
        self.out.release()
        cv2.destroyAllWindows()

        # Print summary statistics
        print(f"Shots attempted: {self.shot_counter}")
        print(f"Shots made: {self.shots_made}")
        print(f"Players tracked: {len(self.unique_ids)}")
        print("Player statistics:")
        for player_id in sorted(self.unique_ids):
            if player_id in [1, 2]:
                shots = self.player_shots.get(player_id, 0)
                makes = self.player_makes.get(player_id, 0)
                pct = (makes / shots * 100) if shots > 0 else 0
                print(f"  Player {player_id}: {makes}/{shots} ({pct:.1f}%)")


def main():
    """Main entry point for the basketball tracking application."""
    import argparse

    parser = argparse.ArgumentParser(description="Basketball Shot Tracking System")
    parser.add_argument(
        "--video", type=str, default="videos/eric.mp4", help="Path to video file"
    )
    parser.add_argument("--camera", type=int, help="Camera ID for live tracking")
    parser.add_argument(
        "--model", type=str, default="models/bball.pt", help="Path to YOLO model"
    )
    parser.add_argument(
        "--output", type=str, default="output_vids", help="Output directory"
    )
    parser.add_argument(
        "--rotate", action="store_true", help="Rotate video 90 degrees clockwise"
    )
    parser.add_argument(
        "--multiple", action="store_true", help="Use multiple player UI mode"
    )
    parser.add_argument(
        "--rim-threshold",
        type=float,
        default=95.0,
        help="Percentage threshold for rim overlap detection",
    )
    args = parser.parse_args()

    # Create and run tracker
    if args.camera is not None:
        tracker = BasketballTracker(
            model_path=args.model,
            camera_id=args.camera,
            output_dir=args.output,
            multiple=args.multiple,  # Pass the multiple parameter
            rim_overlap_threshold=args.rim_threshold,  # Pass the rim threshold parameter
        )
    else:
        tracker = BasketballTracker(
            model_path=args.model,
            video_path=args.video,
            output_dir=args.output,
            multiple=args.multiple,  # Pass the multiple parameter
            rim_overlap_threshold=args.rim_threshold,  # Pass the rim threshold parameter
        )

    tracker.run(rotate_video=args.rotate)


if __name__ == "__main__":
    main()
