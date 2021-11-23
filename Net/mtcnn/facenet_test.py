import cv2
from facenet_pytorch import MTCNN
import numpy as np
import time


class FaceDetector(object):
    '''
    bounding box
    posibility
    markpoint
    '''

    def __init__(self, mtcnn):
        self.mtcnn = mtcnn

    def draw(self, frame, boxes, probs, landmarks):
        '''
        Draw bounding box,probs and landmarks
        Args:
            boxes:
            probs:
            landmarks:

        Returns:

        '''
        for box, prob, ld in zip(boxes, probs, landmarks):
            # draw rectangle
            # print('rec')
            cv2.rectangle(frame,
                          (box[0], box[1]),
                          (box[2], box[3]),
                          (0, 0, 255),
                          thickness=2)
            # print('rec1')
            # show probability
            cv2.putText(frame, str(prob), (box[2], box[3]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1, cv2.LINE_AA)
            # print('rec2')
            # Draw landmarks
            cv2.circle(frame, tuple(ld[0]), 2, (0, 0, 255), 2)
            cv2.circle(frame, tuple(ld[1]), 2, (0, 0, 255), 2)
            cv2.circle(frame, tuple(ld[2]), 2, (0, 0, 255), 2)
            cv2.circle(frame, tuple(ld[3]), 2, (0, 0, 255), 2)
            cv2.circle(frame, tuple(ld[4]), 2, (0, 0, 255), 2)
            # print('rec3')
            return frame

    def run(self):
        cap = cv2.VideoCapture(0)
        num_frames = 1
        while True:
            start = time.time()
            ret, frame = cap.read()
            try:
                boxes, probs, landmarks = self.mtcnn.detect(frame, landmarks=True)
                self.draw(frame, boxes, probs, landmarks)
            except Exception as e:
                # print(e)
                pass
            end = time.time()
            seconds = end - start
            fps = num_frames / seconds
            cv2.putText(frame, "fps: " + str(round(fps)), (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255))
            # src = cv2.resize(frame, (frame_width // 2, frame_height // 2), interpolation=cv2.INTER_CUBIC)
            cv2.imshow('face detector', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()


mtcnn = MTCNN()
fcd = FaceDetector(mtcnn)
fcd.run()
