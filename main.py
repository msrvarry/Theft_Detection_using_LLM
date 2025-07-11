
import cv2
from ultralytics import YOLO
import google.generativeai as genai
import os

# === CONFIG ===
VIDEO_PATH = "sample_video2.mp4"  # Path to your video file
API_KEY =   # Replace this with your actual Gemini API key
CONFIDENCE_THRESHOLD = 0.5
FRAME_INTERVAL = 5  # Analyze every 5th frame

# === Gemini Setup ===
genai.configure(api_key=API_KEY)
gemini_model = genai.GenerativeModel("gemini-1.5-flash")

# Create temp directory to store suspicious frames
os.makedirs("gemini_check_frames", exist_ok=True)

# Load YOLO model
model = YOLO("yolov8n.pt")

# Gemini function to analyze frame
def gemini_theft_check(image_path):
    prompt = (
        "This is a frame from a surveillance video. "
        "Tell me if there is any theft or suspicious action like stealing, hiding, grabbing, or robbery. "
        "If yes, explain briefly. Otherwise, say 'No theft'."
    )
    try:
        with open(image_path, "rb") as f:
            img_data = f.read()
        response = gemini_model.generate_content([
            prompt,
            {"mime_type": "image/jpeg", "data": img_data}
        ])
        return response.text.lower()
    except Exception as e:
        print("[Gemini Error]", e)
        return ""

# Start processing video
cap = cv2.VideoCapture(VIDEO_PATH)
frame_idx = 0
theft_detected = False

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame_idx += 1

    results = model(frame)[0]
    objects_found = False

    # Draw bounding boxes
    for box in results.boxes:
        conf = float(box.conf[0])
        if conf >= CONFIDENCE_THRESHOLD:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls_id = int(box.cls[0])
            label = results.names[cls_id]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            objects_found = True

    # Analyze with Gemini if no confident objects are found
    if frame_idx % FRAME_INTERVAL == 0 and not objects_found:
        temp_path = f"gemini_check_frames/frame_{frame_idx}.jpg"
        cv2.imwrite(temp_path, frame)
        print(f"[Gemini] Analyzing frame {frame_idx}...")
        response = gemini_theft_check(temp_path)
        print("[Gemini Response]", response)

        if any(word in response for word in ["theft", "stealing", "robbery", "suspicious"]):
            theft_detected = True
        else:
            theft_detected = False

    # Show theft alert
    if theft_detected:
        cv2.putText(frame, "🚨 THEFT DETECTED 🚨", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

    # Show frame
    cv2.imshow("Theft Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
