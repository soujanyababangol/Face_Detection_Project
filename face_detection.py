import tensorflow as tf
import cv2
import numpy as np
import os

MODEL_PATH = "model/frozen_inference_graph_face.pb"
INPUT_FOLDER = "image_input"
OUTPUT_FOLDER = "image_output"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)
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

    for image_file in os.listdir(INPUT_FOLDER):
        if not image_file.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        image_path = os.path.join(INPUT_FOLDER, image_file)
        image = cv2.imread(image_path)

        if image is None:
            continue
        height, width, _ = image.shape
        input_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        input_image = np.expand_dims(input_image, axis=0)
        boxes, scores, num = session.run([detection_boxes, detection_scores, num_detections],feed_dict={image_tensor: input_image})
        face_count = 0
        base_name = os.path.splitext(image_file)[0]

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
            face = image[top:bottom, left:right]
            if face.size == 0:
                continue
            face_count += 1
            total_faces += 1
            output_path = os.path.join(OUTPUT_FOLDER,f"{base_name}_face_{face_count}.jpg")
            cv2.imwrite(output_path,face,[cv2.IMWRITE_JPEG_QUALITY, 100])
            print(f"{image_file} - Face {face_count} saved "f"(confidence: {confidence:.2f})")
        print(f"{image_file}: {face_count} faces detected")

print("\nTotal faces detected:", total_faces)
print("Face detection and cropping completed.")

