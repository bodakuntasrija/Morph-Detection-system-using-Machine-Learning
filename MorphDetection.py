import os
import zipfile
import cv2
import numpy as np
import pickle
import tkinter as tk
from tkinter import filedialog
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing import image

# ========================== FILE PATHS ==========================
zip_file_path = "C:/Users/yuvat/Desktop/minor 2/Dataset.zip"
dataset_path = "./dataset"
pca_path = "pca_model.pkl"
model_path = "trained_model.pkl"
output_path = "output_video.mp4"

# ========================== EXTRACT ZIP ==========================
os.makedirs(dataset_path, exist_ok=True)
if not os.path.exists(os.path.join(dataset_path, "Dataset")):
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(dataset_path)

dataset_subfolder = os.path.join(dataset_path, "Dataset")
original_folder = os.path.join(dataset_subfolder, "original_videos")
morphed_folder = os.path.join(dataset_subfolder, "morphed_videos")

# ========================== LOAD VGG16 MODEL ==========================
cnn_model = VGG16(weights='imagenet', include_top=False, pooling='avg')

def extract_cnn_features(face):
    try:
        face_resized = cv2.resize(face, (224, 224))
        img_array = image.img_to_array(face_resized)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)
        features = cnn_model.predict(img_array, verbose=0)
        return features.flatten()
    except:
        return None

def extract_frames(video_path, frame_rate=5):
    cap = cv2.VideoCapture(video_path)
    frames, frame_count = [], 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % frame_rate == 0:
            frames.append(frame)
        frame_count += 1
    cap.release()
    return frames

# ========================== TRAIN PCA + SVM IF NEEDED ==========================
if not os.path.exists(pca_path) or not os.path.exists(model_path):
    def is_valid_file(filename, prefix):
        return filename.startswith(prefix) and filename.lower().endswith((".mp4", ".avi", ".mkv"))

    X_train, y_train = [], []
    for folder, label, prefix in [(original_folder, 0, "original_"), (morphed_folder, 1, "morphed_")]:
        for fname in os.listdir(folder):
            if is_valid_file(fname, prefix):
                frames = extract_frames(os.path.join(folder, fname))
                for frame in frames:
                    features = extract_cnn_features(frame)
                    if features is not None:
                        X_train.append(features)
                        y_train.append(label)

    X_train = np.array(X_train)
    pca = PCA(n_components=256)
    X_train_pca = pca.fit_transform(X_train)
    model = SVC(kernel='linear', probability=True)
    model.fit(X_train_pca, y_train)

    with open(pca_path, 'wb') as f: pickle.dump(pca, f)
    with open(model_path, 'wb') as f: pickle.dump(model, f)
else:
    with open(pca_path, 'rb') as f: pca = pickle.load(f)
    with open(model_path, 'rb') as f: model = pickle.load(f)

# ========================== FACE DETECTION ==========================
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def detect_faces(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

def crop_faces(frame, faces):
    return [frame[y:y+h, x:x+w] for (x, y, w, h) in faces]

def classify_frame_faces(frame):
    faces = detect_faces(frame)
    if len(faces) == 0:
        return "Unknown"
    predictions = []
    for face in crop_faces(frame, faces):
        features = extract_cnn_features(face)
        if features is not None:
            pca_features = pca.transform([features])
            probs = model.predict_proba(pca_features)[0]
            pred = np.argmax(probs)
            predictions.append("Original" if pred == 0 else "Morphed")
    return max(set(predictions), key=predictions.count) if predictions else "Unknown"

# ========================== PROCESS VIDEO ==========================
def process_video(video_path, frame_rate=5):
    frames = extract_frames(video_path, frame_rate)
    frame_results = []

    for frame_idx, frame in enumerate(frames):
        faces = detect_faces(frame)
        print(f"Detected {len(faces)} faces in frame {frame_idx}.")

        if len(faces) == 0:
            print(f"❌ No faces detected in frame {frame_idx}.")
            frame_results.append("Unknown")
            continue

        cropped_faces = crop_faces(frame, faces)
        face_labels = []

        for face_idx, face in enumerate(cropped_faces):
            try:
                features = extract_cnn_features(face)
                print(f"Extracted features: {features.shape}")

                pca_features = pca.transform([features])
                probs = model.predict_proba(pca_features)[0]
                pred = np.argmax(probs)
                result = "Original" if pred == 0 else "Morphed"
                face_labels.append(result)

                print(f"✅ Frame {frame_idx}, Face {face_idx}: {result}")
            except Exception as e:
                print(f"⚠ Error processing face {face_idx} in frame {frame_idx}: {e}")
                face_labels.append("Unknown")

        # Majority vote for frame-level classification
        label_counts = {"Original": 0, "Morphed": 0, "Unknown": 0}
        for label in face_labels:
            label_counts[label] += 1

        if label_counts["Morphed"] > label_counts["Original"]:
            frame_label = "Morphed"
        elif label_counts["Original"] > label_counts["Morphed"]:
            frame_label = "Original"
        else:
            frame_label = "Unknown"

        print(f"✅ Final classification for frame {frame_idx}: {frame_label}")
        frame_results.append(frame_label)

    total_frames = len(frame_results)
    original_count = sum(1 for label in frame_results if label == "Original")
    original_percentage = (original_count / total_frames) * 100 if total_frames > 0 else 0

    if original_percentage > 55:
        video_label = "Original"
    else:
        video_label = "Morphed"

    print(f"✅ Percentage of Original frames: {original_percentage:.2f}%")
    print(f"✅ Final classification for video: {video_label}")
    return video_label, frame_results, any(label != "Unknown" for label in frame_results)
def visualize_results(video_path, results, output_path, frame_rate=5, final_label="Unknown"):
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(fps // frame_rate)

    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), 20.0, (width, height))

    # Set color based on final classification
    if final_label == "Morphed":
        label_text = "MORPHED VIDEO"
        color = (0, 0, 255)  # Red
    elif final_label == "Original":
        label_text = "ORIGINAL VIDEO"
        color = (0, 255, 0)  # Green
    else:
        label_text = "UNKNOWN"
        color = (255, 255, 255)  # White

    frame_idx = 0
    success, frame = cap.read()

    while success:
        # Add label text (video-level classification)
        cv2.putText(frame, label_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

        # Only detect and draw face rectangles every N frames
        if frame_idx % frame_interval == 0:
            faces = detect_faces(frame)
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame, final_label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        out.write(frame)
        success, frame = cap.read()
        frame_idx += 1

    cap.release()
    out.release()

# ========================== GUI ==========================
def select_video():
    video_path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4 *.avi *.mkv")])
    if video_path:
        result_label.config(text="⏳ Processing video...")
        root.update()

        label, frame_results, face_detected = process_video(video_path)

        if face_detected:
            visualize_results(video_path, frame_results, output_path, final_label=label)
            result_label.config(text=f"✅ Final Classification: {label}")
            show_output_video()
        else:
            result_label.config(text="⚠ No face detected in the video. Skipping visualization.")

def show_output_video():
    cap = cv2.VideoCapture(output_path)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imshow("Output Video", frame)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

# ========================== RUN GUI ==========================
root = tk.Tk()
root.title("Face Morph Detection")
root.geometry("400x200")
root.configure(bg="lightblue")

label = tk.Label(root, text="Upload a video to check if it's morphed", bg="lightblue", font=("Arial", 12))
label.pack(pady=20)

upload_button = tk.Button(root, text="Select Video", command=select_video, font=("Arial", 12), bg="skyblue")
upload_button.pack(pady=10)

result_label = tk.Label(root, text="", bg="lightblue", font=("Arial", 12, "bold"))
result_label.pack(pady=20)

root.mainloop()
