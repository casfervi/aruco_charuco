# ArUco and ChArUco Utilities

A small collection of Python utilities for generating ArUco markers, generating ChArUco boards, and estimating the 3D position and orientation of ArUco markers with OpenCV.

The project can be used as a starting point for computer vision experiments involving:

- fiducial marker generation;
- ArUco marker detection;
- marker pose estimation;
- visualization of the X, Y, and Z axes;
- ChArUco board generation;
- preparation of printable patterns for intrinsic camera calibration.

## Project structure

```text
.
├── aruco_generation.py
├── aruco_pose_xyz.py
├── charuco_generation.py
├── markers/
└── README.md
```

## Files

### `aruco_generation.py`

Generates one or more ArUco marker images and saves them to an output directory.

The number of markers is provided through the `--num-markers` command-line argument. The updated script also supports:

- selecting the first marker ID;
- selecting the marker image size in pixels;
- selecting the ArUco dictionary;
- selecting the output directory;
- optionally displaying each generated marker.

The script validates the requested ID range against the capacity of the selected dictionary.

> Attribution: `aruco_generation.py` was obtained from or adapted from the public [JiyaPandey/aruco](https://github.com/JiyaPandey/aruco) repository. That project provides examples of ArUco marker generation, detection, and pose estimation using Python, OpenCV, and NumPy.

### `aruco_pose_xyz.py`

Detects ArUco markers from a webcam or video file and estimates the pose of each marker.

The script can display:

- marker ID;
- marker position as `X`, `Y`, and `Z` relative to the camera coordinate system;
- approximate `roll`, `pitch`, and `yaw` orientation;
- the 3D coordinate axes projected over the marker;
- optional frame-by-frame pose data exported to CSV.

For metric `X`, `Y`, and `Z` values to be reliable, the script should receive:

1. the correct physical marker size;
2. the intrinsic camera matrix;
3. the lens distortion coefficients obtained from camera calibration.

Without intrinsic calibration, the script may use approximate camera parameters for demonstration purposes. In that mode, the resulting coordinates should not be treated as precise measurements.

### `charuco_generation.py`

Generates a printable ChArUco board with configurable physical dimensions.

A ChArUco board combines:

- a chessboard pattern;
- uniquely identifiable ArUco markers;
- known square and marker dimensions.

ChArUco boards are useful for intrinsic camera calibration because their chessboard intersections can be detected with subpixel accuracy while the embedded ArUco markers make the board easier to identify under partial visibility.

### `markers/`

Default directory used to store generated marker images and ChArUco boards.

Example:

```text
markers/
├── marker_0.png
├── marker_1.png
├── marker_2.png
├── marker_3.png
└── charuco_board.png
```

## Requirements

- Python 3.9 or later;
- OpenCV with the `aruco` module;
- NumPy.

Recommended installation:

```bash
pip install opencv-contrib-python numpy
```

Avoid keeping incompatible versions of `opencv-python` and `opencv-contrib-python` in the same environment because they may conflict over the `cv2` package.

Verify the installation with:

```bash
python -c "import cv2; print(cv2.__version__); print(hasattr(cv2, 'aruco')); print(hasattr(cv2.aruco, 'ArucoDetector'))"
```

The final two values should normally be:

```text
True
True
```

## Virtual environment

### Windows

```cmd
python -m venv aruco
aruco\Scripts\activate
pip install opencv-contrib-python numpy
```

### Linux or macOS

```bash
python3 -m venv aruco
source aruco/bin/activate
pip install opencv-contrib-python numpy
```

## Usage

Use the built-in help to inspect the arguments supported by each script:

```bash
python aruco_generation.py --help
python aruco_pose_xyz.py --help
python charuco_generation.py --help
```

## Generating ArUco markers

### Generate the default four markers

```bash
python aruco_generation.py
```

This generates IDs `0` through `3` using the default dictionary and writes the files to `markers/`.

### Select the number of generated marker images

Generate ten markers:

```bash
python aruco_generation.py --num-markers 10
```

### Select the starting ID

Generate five markers beginning with ID 10:

```bash
python aruco_generation.py \
  --num-markers 5 \
  --start-id 10
```

The generated files will be:

```text
marker_10.png
marker_11.png
marker_12.png
marker_13.png
marker_14.png
```

### Select the marker image size

Generate four 800 x 800 pixel markers:

```bash
python aruco_generation.py \
  --num-markers 4 \
  --marker-size 800
```

`--marker-size` controls the image resolution in pixels. It does not define the physical printed size in millimeters.

### Select the dictionary

```bash
python aruco_generation.py \
  --num-markers 4 \
  --dictionary DICT_5X5_250
```

The default dictionary is:

```text
DICT_5X5_250
```

The same dictionary must be used when generating and detecting markers. For example, marker ID `2` in `DICT_5X5_250` is not the same binary pattern as marker ID `2` in `DICT_4X4_50`.

### Select the output directory

```bash
python aruco_generation.py \
  --num-markers 4 \
  --output-dir generated_markers
```

### Display each generated marker

```bash
python aruco_generation.py \
  --num-markers 4 \
  --show
```

When `--show` is enabled, press a key to continue to the next marker or press `Esc` to stop generation.

### Combined example

```bash
python aruco_generation.py \
  --num-markers 4 \
  --start-id 20 \
  --marker-size 600 \
  --dictionary DICT_5X5_250 \
  --output-dir markers \
  --show
```

## Detecting markers and estimating XYZ pose

### Webcam

For a marker whose external black square measures 20 mm:

```bash
python aruco_pose_xyz.py \
  --camera 0 \
  --marker-size-mm 20
```

### Video file

```bash
python aruco_pose_xyz.py \
  --video video.mp4 \
  --marker-size-mm 20
```

### Select the dictionary

```bash
python aruco_pose_xyz.py \
  --camera 0 \
  --marker-size-mm 20 \
  --dictionary DICT_5X5_250
```

### Export poses to CSV

```bash
python aruco_pose_xyz.py \
  --camera 0 \
  --marker-size-mm 20 \
  --dictionary DICT_5X5_250 \
  --csv poses.csv
```

The CSV may contain fields such as:

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
python aruco_pose_xyz.py \
  --camera 0 \
  --marker-size-mm 20 \
  --calibration camera_calibration.npz
```

The NPZ file is expected to contain:

```python
camera_matrix
dist_coeffs
```

## Camera coordinate system

In the OpenCV camera coordinate convention used by the pose-estimation script:

- positive `X` points approximately to the camera's right;
- positive `Y` points approximately downward in the image;
- positive `Z` points forward from the camera into the scene.

The projected axes are normally displayed as:

- red: X axis;
- green: Y axis;
- blue: Z axis.

The reported pose is relative to the camera coordinate system, not necessarily to a global coordinate system for the experiment.

## Generating a ChArUco board

Run:

```bash
python charuco_generation.py
```

Use the script arguments or configuration values to define:

- number of squares in the X and Y directions;
- physical square size;
- physical ArUco marker size;
- ArUco dictionary;
- output image resolution;
- output path.

When printing the board:

- print at 100% scale;
- disable “fit to page” or automatic scaling;
- mount the print on a rigid, flat surface;
- verify the printed square dimensions with a ruler or caliper;
- use the same dictionary during generation and detection.

## Intrinsic camera calibration with ChArUco

Generating the board does not calibrate the camera by itself. Calibration requires multiple images of the same board from different viewpoints.

A useful calibration dataset should include:

- a front-facing view;
- horizontal and vertical board tilts;
- different camera-to-board distances;
- the board near the image corners;
- partial board views containing enough detected corners;
- sharp images without motion blur.

The calibration process should generate:

- `camera_matrix`, containing focal lengths and the optical center;
- `dist_coeffs`, containing lens distortion coefficients;
- a reprojection error used to assess calibration quality.

Use the same camera resolution, focus, and zoom during calibration and subsequent pose estimation. Meaningful optical changes require a new calibration.

## Accuracy considerations

Successful marker detection does not guarantee accurate metric measurements. Pose quality depends on:

- intrinsic camera calibration;
- correct physical marker dimensions;
- flat and accurately scaled printing;
- camera resolution and focus;
- motion blur;
- lighting consistency;
- reflections;
- apparent marker size in the image;
- correct dictionary selection;
- camera stability.

Validate the complete system with known distances or objects that were not used directly during calibration.

## References

### ArUco code reference

- [JiyaPandey/aruco](https://github.com/JiyaPandey/aruco): reference and source/adaptation basis for `aruco_generation.py`. The repository includes examples of ArUco marker generation, detection, and pose estimation using Python, OpenCV, and NumPy.

### OpenCV documentation

- [Detection of ArUco Markers](https://docs.opencv.org/4.13.0/d5/dae/tutorial_aruco_detection.html)
- [Detection of ArUco Boards](https://docs.opencv.org/4.13.0/db/da9/tutorial_aruco_board_detection.html)
- [Detection of ChArUco Boards](https://docs.opencv.org/4.13.0/df/d4a/tutorial_charuco_detection.html)
- [Calibration with ArUco and ChArUco](https://docs.opencv.org/4.13.0/da/d13/tutorial_aruco_calibration.html)
- [Camera Calibration with OpenCV](https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html)

The OpenCV documentation describes ChArUco board creation and detection and recommends ChArUco corners for calibration tasks that require higher corner accuracy than isolated ArUco markers.

## Attribution and licensing note

`aruco_generation.py` was obtained from or adapted from the public [JiyaPandey/aruco](https://github.com/JiyaPandey/aruco) repository. ChArUco-related development was informed by the official OpenCV documentation listed above.

Before redistributing third-party code, review the source repository for its current license and usage terms. If no explicit license is provided, do not assume broad permission to redistribute or relicense the original code.
