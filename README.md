
🤖 Facial Recognition Attendance System

A hassle-free machine-learning-powered attendance system built using **Python**, **OpenCV**, and **face_recognition**. This project captures real-time webcam video, recognizes known faces, and logs attendance to a dated CSV file.

## 📌 Features

- 🧠 Real-time facial recognition using webcam
- 🗂️ CSV logging of names and time of attendance
- 🧾 Loads and encodes multiple known faces from a folder
- 🖼️ Easy to expand with additional face images

## 🚀 How It Works

1. The system loads images from the `faces/` directory.
2. It encodes all known faces at startup.
3. When a known face appears in front of the webcam, it:
   - Displays the name on the video feed
   - Logs the name and time to a CSV file named with the current date
