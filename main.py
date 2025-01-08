import os
import cv2
import numpy as np
import tensorflow as tf
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.image import Image
from kivy.uix.label import Label
from kivy.graphics.texture import Texture
import pygame
from kivy.uix.camera import Camera
from kivy.uix.button import Button

def detect_objects(image_path, model_path="best-fp16.tflite", threshold=0.5):
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


class ObjectDetectionApp(App):
    def build(self):

        self.layout = BoxLayout(orientation="vertical")

        self.camera = Camera(play=True, resolution=(640, 480))
        self.layout.add_widget(self.camera)
        self.image = Image(source="test3.jpg")  
        self.layout.add_widget(self.image)
        

        self.capture_button = Button(text="Detect Objects")
        self.capture_button.bind(on_press=self.capture_frame)
        self.layout.add_widget(self.capture_button)

        return self.layout

    def perform_detection(self, image_path):

        detections, original_image = detect_objects(image_path)

        for detection in detections:
            x1, y1, x2, y2 = detection["bounding_box"]
            cv2.rectangle(original_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
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


        detected_image_path = "detected_image.jpg"
        cv2.imwrite(detected_image_path, original_image)

        self.image.source = detected_image_path
        self.image.reload() 

        var = " "
        for dic in detections:
            var += str(dic["class_id"])+"  "
        self.result_label.text = f"Objects Detected: {len(detections)}"+var

    def capture_frame(self, instance):
        # Capture the current frame from the camera
        texture = self.camera.texture
        frame = np.frombuffer(texture.pixels, np.uint8).reshape(texture.height, texture.width, 4)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)

        # Run object detection
        detections = detect_objects(frame)

        # Process detection results
        if detections:
            self.result_label.text = f"{len(detections)} objects detected"
            for detection in detections:
                print(detection)
        else:
            self.result_label.text = "No objects detected"



if __name__ == "__main__":
    ObjectDetectionApp().run()
