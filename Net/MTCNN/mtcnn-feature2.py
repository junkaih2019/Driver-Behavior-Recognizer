import cv2
from mtcnn_cv2 import MTCNN
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

    def _draw(self, frame, box, prob, ld):
        '''
        Draw bounding box,probs and landmarks
        Args:
            boxes:
            probs:
            landmarks:

        Returns:

        '''
        # for box,prob,ld in zip(boxes,probs,landmarks):
        # draw rectangle
        # print('rec')
        # face rectangle
        cv2.rectangle(frame,
                      (box[0], box[1]),
                      (box[0] + box[2], box[1] + box[3]),
                      (0, 0, 255),
                      thickness=2)
        # phone box
        phone_bia = (int)(0.7 * (max(box[2],box[3])))
        cv2.rectangle(frame,
                      (max(0, ld['nose'][0]-phone_bia), max(0, ld['nose'][1]-phone_bia)),
                      (min(frame.shape[1], ld['nose'][0]+phone_bia),
                       min(frame.shape[0], ld['nose'][1]+phone_bia)),
                      (0, 155, 255),
                      thickness=2)
        # print('rec1')
        # show probability
        cv2.putText(frame, str(prob), (box[0], box[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        # print('rec2')
        # Draw landmarks
        cv2.circle(frame, (ld['left_eye']),2,(0,0,255),2)
        cv2.circle(frame, (ld['right_eye']),2,(0,0,255),2)
        cv2.circle(frame, (ld['nose']),2,(0,0,255),2)
        cv2.circle(frame, (ld['mouth_left']),2,(0,0,255),2)
        cv2.circle(frame, (ld['mouth_right']),2,(0,0,255),2)

        # eye box
        gapEx = (int)(0.3 * (ld['left_eye'][0] - ld['right_eye'][0]))
        gapEy = (int)(0.6 * (ld['nose'][1] - ld['left_eye'][1]))
        eye_bia = (int)(max(gapEx,gapEy))
        cv2.rectangle(frame,
                      (ld['left_eye'][0] - eye_bia, ld['left_eye'][1] + eye_bia),
                      (ld['left_eye'][0] + eye_bia, ld['left_eye'][1] - eye_bia),
                      (0, 155, 255),
                      thickness=2)

        # mouse box
        ctr_x = (int)(ld['mouth_left'][0] + 0.5 * abs((ld['mouth_left'][0] - ld['mouth_right'][0])))
        ctr_y = (int)(ld['mouth_left'][1] + 0.5 * abs((ld['mouth_left'][1] - ld['mouth_right'][1])))
        cv2.circle(frame, (ctr_x,ctr_y), 2, (0, 155, 255), 2)
        mouse_bia =(int)(0.75*(ctr_y - ld['nose'][1]))
        # print((ctr_x - mouse_bia, ctr_y - mouse_bia),
        #      (ctr_x + mouse_bia, ctr_y + mouse_bia))
        cv2.rectangle(frame,
                      (ctr_x - mouse_bia, ctr_y - mouse_bia),
                      (ctr_x + mouse_bia, ctr_y + mouse_bia),
                      (0, 155, 255),
                      thickness=2)

        # print('rec3')
        return frame

    def run(self):
        cap = cv2.VideoCapture(1)
        num_frames = 1
        while True:
            start = time.time()
            ret, frame = cap.read()
            try:
                results = self.mtcnn.detect_faces(frame)
                boxes = results[0]['box']
                probs = results[0]['confidence']
                ld = results[0]['keypoints']
                self._draw(frame, boxes, probs, ld)
            except Exception as e:
                print(e)
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



