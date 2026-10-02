# ArUco, ChArUco, and Fiducial Tracking Utilities

A collection of Python utilities for generating ArUco markers, creating ChArUco calibration boards, estimating marker pose, and analyzing ArUco or AprilTag motion in recorded video with OpenCV.

The repository can be used as a starting point for projects involving:

- ArUco marker generation;
- ChArUco board generation;
- ArUco and AprilTag detection;
- marker pose estimation;
- visualization of the X, Y, and Z axes;
- displacement tracking in pixels or millimeters;
- intrinsic camera calibration workflows;
- interactive video analysis and dashboard export.

---

## Project Structure

```text
.
├── aruco_generation.py
├── aruco_pose_xyz.py
├── charuco_generation.py
├── fiducial_single_video.py
├── markers/
├── videos/                     # suggested folder for test videos
├── camera_calibration.npz      # optional
└── README.md
```

---

## Utilities

### `aruco_generation.py`

Generates one or more ArUco marker images and saves them to an output directory.

Main options include:

- number of markers;
- first marker ID;
- marker image size in pixels;
- ArUco dictionary;
- output directory;
- optional marker preview.

The script validates the requested ID range against the capacity of the selected dictionary.

> **Attribution:** `aruco_generation.py` was obtained from or adapted from the public [JiyaPandey/aruco](https://github.com/JiyaPandey/aruco) repository, which provides examples of ArUco marker generation, detection, and pose estimation using Python, OpenCV, and NumPy.

### `aruco_pose_xyz.py`

Detects markers from a webcam or video file and estimates the pose of each marker.

The script can display or export:

- marker ID;
- X, Y, and Z position relative to the camera;
- approximate roll, pitch, and yaw;
- projected 3D coordinate axes;
- optional frame-by-frame pose data in CSV format.

Reliable metric results require:

- the correct physical marker size;
- a calibrated camera matrix;
- lens-distortion coefficients obtained from intrinsic camera calibration.

Without intrinsic calibration, approximate camera parameters may be used for demonstration, but the resulting coordinates should not be treated as precise measurements.

### `charuco_generation.py`

Generates a printable ChArUco board with configurable physical dimensions.

A ChArUco board combines:

- a chessboard pattern;
- uniquely identifiable ArUco markers;
- known square and marker dimensions.

ChArUco boards are useful for intrinsic camera calibration because chessboard intersections can be detected with subpixel accuracy, while embedded markers support identification under partial visibility.

### `fiducial_single_video.py`

Analyzes one recorded video containing an ArUco or AprilTag marker and presents the tracking results in a resizable visual dashboard.

This utility is intended for displacement analysis rather than only pose visualization. It preprocesses the complete video, tracks one selected marker ID, and then opens an interactive player containing:

- a large video panel with the detected marker outlined;
- an accumulated XY displacement plane;
- a whole-video graph of `dX` and `dY`;
- a detection timeline;
- current displacement and net displacement;
- accumulated path length;
- frame-by-frame playback and seeking;
- dashboard video export.

#### Supported marker families

The script supports the ArUco dictionaries included in the repository workflow and the following AprilTag dictionaries exposed by OpenCV:

```text
DICT_APRILTAG_16h5
DICT_APRILTAG_25h9
DICT_APRILTAG_36h10
DICT_APRILTAG_36h11
```

The selected dictionary must match the physical marker in the video.

#### Pixel and millimeter displacement

Without `--marker-size-mm`, displacement is calculated from the marker center and reported in pixels.

With `--marker-size-mm`, the script estimates marker pose using `cv2.solvePnP` with `SOLVEPNP_IPPE_SQUARE` and reports displacement in millimeters.

For accurate metric results, also provide a calibration file containing:

```text
camera_matrix
dist_coeffs
```

#### Dashboard layout

The dashboard includes:

1. **Header** with the video name, dictionary, marker ID, playback state, frame number, timestamp, and detection status.
2. **Video panel** with the original aspect ratio preserved and the tracked marker highlighted.
3. **Accumulated XY plane** showing the full trajectory and the path reached at the current frame.
4. **Whole-video displacement graph** showing `dX` and `dY` for the complete recording.
5. **Detection timeline** indicating valid and missing detections over time.

The interactive window can be resized, and the dashboard adapts to the current window dimensions.

#### Exporting the dashboard

Press `S` during playback to export the complete dashboard as an MP4 file next to the input video.

The dashboard can also be exported directly from the command line with `--save`:

```bash
python fiducial_single_video.py videos/test_video.mp4   --dictionary DICT_APRILTAG_36h11   --save output_dashboard.mp4
```

Supported output extensions are `.mp4` and `.avi`.

#### Example:

https://github.com/user-attachments/assets/0b4c89be-c6e0-4b6a-95f9-7fa396a7c3a8


---

## Output Directories

### `markers/`

Default directory for generated marker images and ChArUco boards.

Example:

```text
markers/
├── marker_0_DICT_5X5_250.png
├── marker_1_DICT_5X5_250.png
├── marker_2_DICT_5X5_250.png
├── marker_3_DICT_5X5_250.png
└── charuco_board.png
```

### `videos/`

Suggested directory for test and example videos.

Example:

```text
videos/
├── aruco_test.mp4
└── apriltag_test.mp4
```

---

## Requirements

- Python 3.9 or later;
- OpenCV with the `aruco` module;
- NumPy.

Install the dependencies with:

```bash
pip install opencv-contrib-python numpy
```

Use `opencv-contrib-python`, rather than only `opencv-python`, because these utilities require `cv2.aruco`.

Avoid keeping incompatible versions of `opencv-python` and `opencv-contrib-python` in the same environment because both packages provide the `cv2` module.

Verify the installation:

```bash
python -c "import cv2; print(cv2.__version__); print(hasattr(cv2, 'aruco')); print(hasattr(cv2.aruco, 'ArucoDetector'))"
```

The final two values should normally be:

```text
True
True
```

---

## Virtual Environment

The suggested virtual-environment name is `aruco`.

### Windows

```bat
python -m venv aruco
aruco\Scriptsctivate
pip install opencv-contrib-python numpy
```

### Linux or macOS

```bash
python3 -m venv aruco
source aruco/bin/activate
pip install opencv-contrib-python numpy
```

---

## Usage Overview

Inspect the arguments supported by each script:

```bash
python aruco_generation.py --help
python aruco_pose_xyz.py --help
python charuco_generation.py --help
python fiducial_single_video.py --help
```

---

## Generating ArUco Markers

### Generate the default markers

```bash
python aruco_generation.py
```

### Select the number of markers

```bash
python aruco_generation.py --num-markers 10
```

### Select the starting ID

```bash
python aruco_generation.py   --num-markers 5   --start-id 10
```

### Select image size, dictionary, and output directory

```bash
python aruco_generation.py   --num-markers 4   --start-id 20   --marker-size 600   --dictionary DICT_5X5_250   --output-dir markers   --show
```

`--marker-size` controls image resolution in pixels. It does not define the physical printed size in millimeters.

The dictionary used during detection must match the dictionary used during generation.

---

## Detecting Markers and Estimating XYZ Pose

### Webcam

```bash
python aruco_pose_xyz.py   --camera 0   --marker-size-mm 20   --dictionary DICT_5X5_250
```

### Video file

```bash
python aruco_pose_xyz.py   --video videos/test_video.mp4   --marker-size-mm 20   --dictionary DICT_5X5_250
```

### Export pose data to CSV

```bash
python aruco_pose_xyz.py   --camera 0   --marker-size-mm 20   --dictionary DICT_5X5_250   --csv poses.csv
```

The CSV may contain:

```text
timestamp_s
frame
marker_id
x_mm
y_mm
z_mm
roll_deg
pitch_deg
yaw_deg
```

### Load intrinsic camera calibration

```bash
python aruco_pose_xyz.py   --camera 0   --marker-size-mm 20   --calibration camera_calibration.npz
```

---

## Single-Video Displacement Dashboard

### ArUco example

```bash
python fiducial_single_video.py videos/aruco_test.mp4   --dictionary DICT_4X4_250   --marker-id 0
```

### AprilTag example

```bash
python fiducial_single_video.py videos/apriltag_test.mp4   --dictionary DICT_APRILTAG_36h11   --marker-id 0
```

If `--marker-id` is omitted, the program selects the lowest marker ID found in the first successful detection.

### Millimeter displacement with calibration

```bash
python fiducial_single_video.py videos/apriltag_test.mp4   --dictionary DICT_APRILTAG_36h11   --marker-id 0   --marker-size-mm 50   --calibration camera_calibration.npz
```

### Full-resolution AprilTag detection

AprilTag detection uses `--detect-scale 0.5` by default. If the marker is small in the image, use:

```bash
python fiducial_single_video.py videos/apriltag_test.mp4   --dictionary DICT_APRILTAG_36h11   --detect-scale 1.0
```

The displayed video and saved coordinates remain associated with the original video resolution. `--detect-scale` changes only the image resolution used by the detector.

### Initial window and export size

```bash
python fiducial_single_video.py videos/apriltag_test.mp4   --dictionary DICT_APRILTAG_36h11   --width 1920   --height 1080
```

### Direct dashboard export

```bash
python fiducial_single_video.py videos/apriltag_test.mp4   --dictionary DICT_APRILTAG_36h11   --save output_dashboard.mp4   --width 1920   --height 1080
```

### Playback controls

```text
SPACE          Pause or resume
A / Left       Move back one frame and pause
D / Right      Move forward one frame and pause
J              Move back approximately one second
L              Move forward approximately one second
Home           Go to the first frame
End            Go to the final frame
S              Export the full dashboard video
Q / Esc        Quit
Frame bar      Seek directly to a frame
```

---

## Camera Coordinate Systems

### Pose-estimation coordinates

For `aruco_pose_xyz.py`, the OpenCV camera-coordinate convention is:

- positive X points approximately to the camera's right;
- positive Y points approximately downward in the image;
- positive Z points forward from the camera into the scene.

The projected axes are normally drawn as:

- red: X;
- green: Y;
- blue: Z.

### Dashboard displacement coordinates

For `fiducial_single_video.py`:

- positive `dX` indicates movement to the right;
- negative `dX` indicates movement to the left;
- positive `dY` indicates movement downward;
- negative `dY` indicates movement upward.

Displacement is measured relative to the first frame in which the selected marker is detected.

---

## Generating a ChArUco Board

Run:

```bash
python charuco_generation.py
```

Use the script arguments or configuration values to define:

- number of squares in the X and Y directions;
- physical square size;
- physical marker size;
- ArUco dictionary;
- output image resolution;
- output path.

When printing the board:

- print at 100% scale;
- disable fit-to-page or automatic scaling;
- mount the print on a rigid, flat surface;
- verify the printed dimensions with a ruler or caliper;
- use the same dictionary during generation and detection.

---

## Intrinsic Camera Calibration with ChArUco

Generating a ChArUco board does not calibrate the camera by itself. Calibration requires multiple images of the same board from different viewpoints.

A useful calibration dataset should include:

- a front-facing view;
- horizontal and vertical tilts;
- different camera-to-board distances;
- views with the board near the image corners;
- partial board views containing enough detected corners;
- sharp images without motion blur.

The calibration process should generate:

- `camera_matrix`, containing focal lengths and the optical center;
- `dist_coeffs`, containing lens-distortion coefficients;
- a reprojection error used to assess calibration quality.

Use the same camera resolution, focus, and zoom during calibration and subsequent measurements.

---

## Accuracy Considerations

Successful marker detection does not guarantee accurate metric measurements. Measurement quality depends on:

- intrinsic camera calibration;
- correct physical marker dimensions;
- flat and accurately scaled printing;
- camera resolution and focus;
- marker size in the image;
- motion blur;
- lighting consistency;
- reflections;
- lens distortion;
- correct dictionary selection;
- camera stability.

Validate the complete measurement system using known distances or objects that were not used directly during camera calibration.

---

## References

### ArUco code reference

- [JiyaPandey/aruco](https://github.com/JiyaPandey/aruco): reference and source or adaptation basis for `aruco_generation.py`.

### OpenCV documentation

- [Detection of ArUco Markers](https://docs.opencv.org/4.13.0/d5/dae/tutorial_aruco_detection.html)
- [Detection of ArUco Boards](https://docs.opencv.org/4.13.0/db/da9/tutorial_aruco_board_detection.html)
- [Detection of ChArUco Boards](https://docs.opencv.org/4.13.0/df/d4a/tutorial_charuco_detection.html)
- [Calibration with ArUco and ChArUco](https://docs.opencv.org/4.13.0/da/d13/tutorial_aruco_calibration.html)
- [Camera Calibration with OpenCV](https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html)

---

## Attribution and Licensing Note

`aruco_generation.py` was obtained from or adapted from the public [JiyaPandey/aruco](https://github.com/JiyaPandey/aruco) repository. ChArUco-related development was informed by the official OpenCV documentation listed above.

Before redistributing third-party code, review the source repository for its current license and usage terms. If no explicit license is provided, do not assume broad permission to redistribute or relicense the original code.
