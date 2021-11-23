import cv2
import time

cap = cv2.VideoCapture(0)
num_frames = 1
while True:
            start = time.time()
            ret, frame = cap.read()
            end = time.time()
            seconds = end - start
            if(seconds):
                fps = num_frames /seconds
            cv2.putText(frame,"fps: " + str(round(fps)),(30,40),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255))
            cv2.imshow('face detector',frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
cap.release()
cv2.destroyAllWindows()