import cv2
from ultralytics import YOLO

# Load model
model = YOLO("models/yolov8n.pt")
cap = cv2.VideoCapture(0)

print("--- CALIBRATION MODE ---")
print("Stand exactly 2.0 meters away.")
print("Press 'q' to quit.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    # Run detection
    results = model(frame, classes=[0], verbose=False)

    for r in results:
        for box in r.boxes:
            # Get bounding box coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            # Calculate height in pixels
            pixel_height = y2 - y1
            
            # Draw box and text
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"Height: {pixel_height} px", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            
            # Print to console for easy reading
            print(f"Detected Height: {pixel_height} pixels")

    cv2.imshow("Calibration", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()