# YOLOv8 Person Detection with Clothing Color Classification

A real-time webcam application that detects people and assigns them labels (enemy, friend, romance target) based on their detected clothing color.

## Features

- **Real-time People Detection**: Uses YOLOv8 for fast and accurate person detection
- **Clothing Color Analysis**: Detects the dominant color in the upper body/shoulder area of detected people
- **Automatic Labeling**: Assigns labels based on clothing color:
  - **Red**: Enemy
  - **Blue/Green**: Friend
  - **Yellow**: Romance Target
  - **Purple**: Enemy
  - **Black**: Enemy

## Installation

1. Clone or navigate to the project directory
2. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

   Note: First-time setup may take a few minutes as it downloads the YOLOv8 model (~100MB)

## Usage

Run the application:
```bash
python main.py
```

### Controls
- **Press 'q'** to quit the application
- The webcam feed will display in a window with:
  - Colored bounding boxes around detected people
  - Labels showing the assigned category and detected color
  - Confidence scores for each detection

### Output Colors
- **Red box**: Enemy (red clothing detected)
- **Green box**: Friend (blue or green clothing detected)
- **Yellow box**: Romance target (yellow clothing detected)
- **Gray box**: Unknown (unclassified)

## How It Works

1. **Detection**: YOLOv8 detects all people in the webcam frame
2. **Color Analysis**: For each detected person, the upper portion of their bounding box is analyzed
3. **HSV Color Space**: Colors are detected using HSV (Hue, Saturation, Value) which is more robust to lighting conditions
4. **Labeling**: The dominant clothing color determines the label assignment

## Color Detection Ranges

The color detection uses HSV color space ranges:
- **Red**: Hue 0-10 and 170-180
- **Blue**: Hue 100-130
- **Green**: Hue 40-80
- **Yellow**: Hue 20-40
- **Purple**: Hue 130-160
- **Black**: Low saturation (0-100) and low value/brightness (0-50)

## Customization

You can modify the label mapping by editing the `label_mapping` dictionary in the `PersonLabelAssigner` class:

```python
self.label_mapping = {
    'red': 'enemy',      # Change this to reassign red
    'blue': 'friend',
    'green': 'friend',
    'yellow': 'romance target',
    'purple': 'enemy'
}
```

Or adjust color detection ranges by modifying the `color_ranges` dictionary.

## System Requirements

- Python 3.8 or higher
- Webcam
- At least 2GB RAM (more for better performance)
- GPU recommended for faster inference (CPU will work but slower)

## Performance Notes

- The script uses YOLOv8 nano model (`yolov8n.pt`) for speed
- For higher accuracy, you can change to `yolov8s.pt` or `yolov8m.pt`
- First run will download the model automatically
- Frame resolution is set to 640x480 - adjust in code if needed

## Troubleshooting

**Webcam not detected**: Ensure your webcam is properly connected and no other application is using it

**Slow performance**: 
- Use a smaller model or reduce frame resolution
## YOLOv8 Person Detection with RPG-style HP/Mana Overlay

Real-time webcam app that detects people, analyzes their upper-body clothing color, and assigns simple RPG-style HP and Mana values which are displayed as bars above each person.

## Features

- Real-time person detection using YOLOv8 (Ultralytics)
- Upper-body color analysis (dominant color) → Mana mapping
- HP derived from size (reference-distance normalization)
- Pending assignment: a person must be visible for `pending_timeout` seconds before HP/Mana are committed
- Persistent per-person attributes across short occlusions (IoU/proximity matching)

## Installation

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Download or ensure the YOLO model exists at `models/yolov8n.pt` (first run will auto-download)

## Usage

Run the main pose/detection script:

```bash
python src/main_pose.py
```

- Press `q` to quit.

## Configuration

The application reads `config.yaml` by default. Key options:

- `smoothing_alpha`: EMA alpha for smoothing HP/Mana (0.0 - 1.0). Higher favors new measurements.
- `ref_distance`: Reference distance in meters used for HP scaling (default 2.0)
- `hp_scale`: Multiplicative calibration for HP values
- `pending_timeout`: Seconds a detection must remain visible before assignment (default 2.5)
- `person_timeout`: Seconds after last seen to forget a person (default 30)

Example `config.yaml` is included in the repo.

## Requirements

- Python 3.8+
- Dependencies (in `requirements.txt`): `ultralytics`, `opencv-python`, `numpy`, `PyYAML`, `torch` (install matches your platform/CUDA if desired)

## How It Works (brief)

1. YOLOv8 detects people and provides tracked boxes.
2. For unmatched detections, the app starts a pending timer (configured by `pending_timeout`).
3. While pending, a gray box and provisional bars/countdown are shown.
4. After the timeout, the upper-body region is analyzed for dominant color and size; HP and Mana are computed and committed.
5. Committed HP/Mana are displayed above the person and persisted until the person is forgotten (`person_timeout`).

## Troubleshooting & Tips

- If HP values are too small/large, tweak `hp_scale` in `config.yaml`.
- Increase `pending_timeout` if people move quickly and you get mistaken assignments.
- Ensure good lighting for reliable color detection.

If you want, I can add an example command to run with a custom config or adjust the default `pending_timeout` for faster assignment.
