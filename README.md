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
- Check your system's GPU availability
- Close other resource-intensive applications

**Color detection not working**:
- Ensure adequate lighting
- Adjust HSV color ranges in the code
- Wear clothing with more saturated colors
