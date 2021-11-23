import cv2
from PIL import Image


class FaceDetector(object):
    def __init__(self, mtcnn):
        self.mtcnn = mtcnn

    def get_3_pos(self,frame,box,ld):
        # face position
        phone_bia = (int)(0.7 * (max(box[2], box[3])))
        phone_pos = ((max(0, ld['nose'][0] - phone_bia),
                      max(0, ld['nose'][1] - phone_bia)),
                    (min(frame.shape[1], ld['nose'][0] + phone_bia),
                     min(frame.shape[0], ld['nose'][1] + phone_bia)),)

        # eye position
        gapEx = (int)(0.3 * abs((ld['left_eye'][0] - ld['right_eye'][0])))
        gapEy = (int)(0.6 * (ld['nose'][1] - ld['left_eye'][1]))
        eye_bia = (int)(max(gapEx, gapEy))
        # print(eye_bia, ld['left_eye'])
        eye_pos = ((ld['left_eye'][0] - eye_bia, ld['left_eye'][1] - eye_bia),
                   (ld['left_eye'][0] + eye_bia, ld['left_eye'][1] + eye_bia))

        # mouse position
        ctr_x = (int)(ld['mouth_left'][0] + 0.5 * abs((ld['mouth_left'][0] - ld['mouth_right'][0])))
        ctr_y = (int)(ld['mouth_left'][1] + 0.5 * abs((ld['mouth_left'][1] - ld['mouth_right'][1])))
        mouse_bia = (int)(0.75 * (ctr_y - ld['nose'][1]))
        mouse_pos = ((ctr_x - mouse_bia, ctr_y - mouse_bia),
                     (ctr_x + mouse_bia, ctr_y + mouse_bia))
        return phone_pos, eye_pos, mouse_pos

    def get_3_pics(self,trans,frame,box,ld):
        phone_pos, eye_pos, mouse_pos = self.get_3_pos(frame, box, ld)
        face_img = frame[phone_pos[0][1]:phone_pos[1][1], phone_pos[0][0]:phone_pos[1][0]]
        mouse_img = frame[mouse_pos[0][1]:mouse_pos[1][1], mouse_pos[0][0]:mouse_pos[1][0]]
        eye_img = frame[eye_pos[0][1]:eye_pos[1][1], eye_pos[0][0]:eye_pos[1][0]]
        # cv2.imshow('eye',eye_img)
        # cv2.waitKey(1)
        eye_img = Image.fromarray(eye_img, mode='RGB')
        # eye_img.show()
        face = trans(Image.fromarray(face_img)).reshape(1, 3, 224, 224)
        mouse = trans(Image.fromarray(mouse_img)).reshape(1, 3, 224, 224)
        eye = trans(eye_img).reshape(1, 3, 224, 224)
        return face, mouse, eye

    def draw(self, frame, box, prob, ld):
        '''
        Draw bounding box,probs and landmarks
        '''

        phone_pos, eye_pos, mouse_pos = self.get_3_pos(frame,box,ld)
        # face box
        cv2.rectangle(frame,
                      (box[0], box[1]),
                      (box[0] + box[2], box[1] + box[3]),
                      (0, 0, 255),
                      thickness=2)
        # phone box
        cv2.rectangle(frame,
                      phone_pos[0],
                      phone_pos[1],
                      (0, 155, 255),
                      thickness=2)
        # eye box
        cv2.rectangle(frame,
                      eye_pos[0],
                      eye_pos[1],
                      (0, 155, 255),
                      thickness=2)
        # mouse box
        cv2.rectangle(frame,
                      mouse_pos[0],
                      mouse_pos[1],
                      (0, 155, 255),
                      thickness=2)
        # show probability
        cv2.putText(frame, str(prob), (box[0], box[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

        # draw landmarks
        cv2.circle(frame, (ld['left_eye']),2,(0,0,255),2)
        cv2.circle(frame, (ld['right_eye']),2,(0,0,255),2)
        cv2.circle(frame, (ld['nose']),2,(0,0,255),2)
        cv2.circle(frame, (ld['mouth_left']),2,(0,0,255),2)
        cv2.circle(frame, (ld['mouth_right']),2,(0,0,255),2)
        return frame

    # def detect(self):




