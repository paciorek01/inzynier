import cv2
import numpy as np
import tensorflow as tf
from gtts import gTTS
import os
import pygame


def detect_objects(image_path, model_path="best-fp16.tflite", threshold=0.3):

    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()


    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()


    image = cv2.imread(image_path)
    original_image = image.copy()  
    image_height, image_width, _ = original_image.shape


    resized_image = cv2.resize(image, (640, 640))  
    input_data = np.expand_dims(resized_image / 255.0, axis=0).astype(np.float32)

    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    output_data = interpreter.get_tensor(output_details[0]['index'])

    detections = []
    for detection in output_data[0]:
        confidence = detection[4]
        if confidence > threshold:
            x_center, y_center, width, height = detection[:4]
            class_probs = detection[5:]  

            class_id = int(np.argmax(class_probs))

            x1 = int((x_center - width / 2) * image_width)
            y1 = int((y_center - height / 2) * image_height)
            x2 = int((x_center + width / 2) * image_width)
            y2 = int((y_center + height / 2) * image_height)

            detections.append({
                "bounding_box": [x1, y1, x2, y2],
                "confidence": confidence,
                "class_id": class_id
            })
    
    idList = []
    for i in detections:
        detectionClassId = i["class_id"]
        if i["class_id"] not in idList:
            idList.append(i["class_id"])
    filtered = []
    for detection in detections:
        detectionClassId = detection["class_id"]
        detectionConfidance = detection["confidence"]
        if detectionClassId in idList:
            filtered.append(detection)
            idList.remove(detectionClassId)
        else:
            test = [x for x in filtered if x["class_id"] == detectionClassId][0]
            if test['confidence'] < detectionConfidance:
                filtered.remove(test) 
                filtered.append(detection)  
    return filtered, original_image

def main():
    image_path = "test5.jpg"  
    model_path = "best-fp16.tflite"  

    
    detections, detected_image = detect_objects(image_path, model_path)

    for detection in detections:
        x1, y1, x2, y2 = detection["bounding_box"]
        cv2.rectangle(detected_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        pygame.mixer.init()
        
        match detection["class_id"]:
            case 0: 
                pygame.mixer.music.load("files/r_159.mp3")
                pygame.mixer.music.play()
            case 1:
                pygame.mixer.music.load("files/r_45.mp3")
                pygame.mixer.music.play()
            case 2:
                pygame.mixer.music.load("files/r_50.mp3")
                pygame.mixer.music.play()
            case 3:
                pygame.mixer.music.load("files/r_72.mp3")
                pygame.mixer.music.play()


        # label = f"Class: {class_id}, Conf: {confidence:.2f}"
        # cv2.putText(detected_image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)


    cv2.imshow("Object Detection", detected_image)
    cv2.waitKey(0)  
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
