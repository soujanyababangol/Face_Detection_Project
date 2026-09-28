import tensorflow as tf
import cv2
import numpy as np
import os

MODEL_PATH = "model/frozen_inference_graph_face.pb"
INPUT_FOLDER = "video_input"
OUTPUT_FOLDER = "video_output"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
for file in os.listdir(OUTPUT_FOLDER):
    file_path = os.path.join(OUTPUT_FOLDER, file)
    if os.path.isfile(file_path):
        os.remove(file_path)
graph = tf.Graph()

with graph.as_default():
    graph_def = tf.compat.v1.GraphDef()
    with tf.io.gfile.GFile(MODEL_PATH, "rb") as file:
        graph_def.ParseFromString(file.read())
    tf.import_graph_def(graph_def, name="")
print("Face detection model loaded.")

with tf.compat.v1.Session(graph=graph) as session:
    image_tensor = graph.get_tensor_by_name("image_tensor:0")
    detection_boxes = graph.get_tensor_by_name("detection_boxes:0")
    detection_scores = graph.get_tensor_by_name("detection_scores:0")
    num_detections = graph.get_tensor_by_name("num_detections:0")
    total_faces = 0
    for video_file in os.listdir(INPUT_FOLDER):
        if not video_file.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
            continue
        video_path = os.path.join(INPUT_FOLDER, video_file)
        video = cv2.VideoCapture(video_path)

        if not video.isOpened():
            print(f"Could not open {video_file}")
            continue
        frame_number = 0
        video_faces = 0
        base_name = os.path.splitext(video_file)[0]
        while True:
            success, frame = video.read()
            if not success:
                break
            frame_number += 1
            height, width, _ = frame.shape
            input_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            input_frame = np.expand_dims(input_frame,axis=0)
            boxes, scores, num = session.run(
                [
                    detection_boxes,
                    detection_scores,
                    num_detections
                ],
                feed_dict={
                    image_tensor: input_frame
                }
            )
            frame_face_count = 0
            for i in range(int(num[0])):
                confidence = scores[0][i]
                if confidence < 0.30:
                    continue
                ymin, xmin, ymax, xmax = boxes[0][i]
                left = max(0, int(xmin * width))
                top = max(0, int(ymin * height))
                right = min(width, int(xmax * width))
                bottom = min(height, int(ymax * height))
                if right <= left or bottom <= top:
                    continue
                face = frame[top:bottom, left:right]
                if face.size == 0:
                    continue
                frame_face_count += 1
                video_faces += 1
                total_faces += 1
                output_path = os.path.join(OUTPUT_FOLDER,f"{base_name}_frame_{frame_number}_face_{frame_face_count}.jpg")

                cv2.imwrite(output_path,face,[cv2.IMWRITE_JPEG_QUALITY, 100])
        video.release()
        print(f"{video_file}: {video_faces} face crops saved")

print("\nTotal face crops saved:", total_faces)
print("Video face detection and cropping completed.")

