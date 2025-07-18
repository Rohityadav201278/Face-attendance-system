
import face_recognition
import cv2
import numpy as np
import csv
import os
from datetime import datetime

print("Starting attendance system...")

# Load known faces
print("🔍 Loading known faces...")
known_face_encodings = []
known_face_names = []

import os

face_dir = os.path.join(os.path.dirname(__file__), "faces")

for filename in os.listdir(face_dir):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        name = os.path.splitext(filename)[0]
        path = os.path.join(face_dir, filename)
        image = face_recognition.load_image_file(path)
        encodings = face_recognition.face_encodings(image)
        if encodings:
            known_face_encodings.append(encodings[0])
            known_face_names.append(name.capitalize())
            print(f" Loaded {name}")
        else:
            print(f"No face found in {filename}, skipping.")

if not known_face_encodings:
    print(" No faces loaded. Exiting...")
    exit()

students = known_face_names.copy()

# Webcam
video_capture = cv2.VideoCapture(0)
if not video_capture.isOpened():
    print(" Webcam not found or permission denied.")
    exit()

print(" Webcam initialized. Press 'q' to quit.")

# Create CSV file
now = datetime.now()
date_str = now.strftime("%Y-%m-%d")
csv_path = f"{date_str}.csv"

with open(csv_path, "w", newline="") as csvfile:
    writer = csv.writer(csvfile)

    while True:
        ret, frame = video_capture.read()
        if not ret:
            print("⚠️ Frame not captured, skipping...")
            continue

        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        face_locations = face_recognition.face_locations(rgb_small)
        face_encodings = face_recognition.face_encodings(rgb_small, face_locations)

        for face_encoding, face_location in zip(face_encodings, face_locations):
            distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            if len(distances) == 0:
                continue

            best_match_idx = np.argmin(distances)
            if distances[best_match_idx] < 0.5:
                name = known_face_names[best_match_idx]

                if name in students:
                    students.remove(name)
                    time_str = datetime.now().strftime("%H:%M:%S")
                    writer.writerow([name, time_str])
                    print(f"📝 Marked present: {name} at {time_str}")

                top, right, bottom, left = [v * 4 for v in face_location]
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                cv2.putText(frame, name + " Present", (left + 6, bottom - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow("Attendance", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

video_capture.release()
cv2.destroyAllWindows()
print(" Attendance saved to:", csv_path)
