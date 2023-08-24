from ultralytics import YOLO
import cv2
import supervision as sv
import numpy as np
from playsound import playsound


START = sv.Point(0, 180)
END = sv.Point(640, 220)

SOURCE = "/home/dax/Escritorio/Proyectos/ComputerVision/Video.mp4"


def verficar_2_carros(detections):
    carros = False
    if len(detections) >= 2:
        if detections.xyxy[0][0] < 300 or detections.xyxy[0][3] > 220:
           carros = True
        if carros:
            if detections.xyxy[1][0] < 300 or detections.xyxy[1][3] > 220:

                playsound("alarma.mp3")
            


def main():
    model = YOLO("yolov8l.pt")
    line_zone = sv.LineZone(start=START, end=END)
    line_zone_annotator = sv.LineZoneAnnotator(thickness=2, text_thickness=1,text_scale=0.5)
    box_annotator = sv.BoxAnnotator(
        thickness=2,
        text_thickness=1,
        text_scale=0.5
    )

    for result in model.track(source=SOURCE, show=False, stream=True):
        frame = result.orig_img

        detections = sv.Detections.from_yolov8(result)
        detections = detections[(detections.class_id == 2) & (detections.area>6000)]    
        verficar_2_carros(detections)
        if result.boxes.id is not None:

            detections.tracker_id = result.boxes.id.cpu().numpy().astype(int)


        


        labels = [f"{model.model.names[class_id]} {confidence:0.2f}"
            for _, _, confidence, class_id, _ in detections
            ]

        frame = box_annotator.annotate(scene=frame, detections=detections, labels=labels)

        line_zone.trigger(detections=detections)
        line_zone_annotator.annotate(frame=frame, line_counter=line_zone)


        cv2.imshow("Puerta", frame)

        if(cv2.waitKey(30) == 27):
            break

if __name__ == "__main__":
    main()