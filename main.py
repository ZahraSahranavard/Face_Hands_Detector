import cv2
import numpy as np
from cvzone.HandTrackingModule import HandDetector
from cvzone.FaceMeshModule import FaceMeshDetector


cap = cv2.VideoCapture(0)
detector_hands = HandDetector(detectionCon=0.5)
meshdetector = FaceMeshDetector()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture image")
        break

    hands, frame = detector_hands.findHands(frame)
    frame, faces = meshdetector.findFaceMesh(frame)

    num_hands = len(hands)
    num_faces = len(faces)
    y_offset = 50

    cv2.putText(frame, "Faces Detected: {0}".format(num_faces), (50, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    y_offset += 40

    if faces:
        for face in faces:
            points = np.array(face)
            hull = cv2.convexHull(points)
            area = cv2.contourArea(hull)
            cv2.putText(frame, "Face Area: {0}".format(int(area)), (50, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            y_offset += 40

    for i, hand in enumerate(hands):
        lmlist = hand["lmList"]
        length, info, frame = detector_hands.findDistance(lmlist[4][:2], lmlist[8][:2], frame)
        color = (0, 255, 0)
        cv2.putText(frame, "Length Hand {0}: {1}".format(i + 1, int(length)), (50, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)


        if length < 50:
            color = (255, 0, 0)
            cv2.putText(frame, "Close! (Hand {0})".format(i+1), (50, y_offset + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 0, 255), 1)

        y_offset += 70

    cv2.imshow("Frame", frame)

    key = cv2.waitKey(5) & 0xFF
    if key == 27:  #Esc
        break

cv2.destroyAllWindows()
cap.release()