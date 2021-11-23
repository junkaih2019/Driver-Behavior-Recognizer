import cv2
import os
import time

from PIL import Image
from mtcnn_cv2 import MTCNN


class CutPic:
    def __init__(self, save_dir, img, face_pos, eye_pos, mouse_pos, ):
        self.save_dir = save_dir
        self.img = img
        self.face_pos = face_pos
        self.eye_pos = eye_pos
        self.mouse_pos = mouse_pos

    def cut(self):
        now_time = int(time.time() * 1000)
        face_dir = self.save_dir + 'face'
        face_path = face_dir + '/face_%d.jpg' % now_time
        if not os.path.exists(face_dir):
            os.makedirs(face_dir)

        eye_dir = self.save_dir + 'eye'
        eye_path = eye_dir + '/eye_%d.jpg' % now_time
        if not os.path.exists(eye_dir):
            os.makedirs(eye_dir)

        mouse_dir = self.save_dir + 'mouse'
        mouse_path = mouse_dir + '/mouse_%d.jpg' % now_time
        if not os.path.exists(mouse_dir):
            os.makedirs(mouse_dir)

        # print(face_pos[0][0], face_pos[1][0], face_pos[0][1], face_pos[1][1])
        face_img = self.img[self.face_pos[0][1]:self.face_pos[1][1], self.face_pos[0][0]:self.face_pos[1][0]]
        # cv2.imshow("face",cv2.cvtColor(face_img, cv2.COLOR_RGB2BGR))
        # cv2.waitKey(0)
        mouse_img = self.img[self.mouse_pos[0][1]:self.mouse_pos[1][1], self.mouse_pos[0][0]:self.mouse_pos[1][0]]
        # cv2.imshow("mouse", cv2.cvtColor(mouse_img, cv2.COLOR_RGB2BGR))
        # cv2.waitKey(0)
        eye_img = self.img[self.eye_pos[0][1]:self.eye_pos[1][1], self.eye_pos[0][0]:self.eye_pos[1][0]]
        # cv2.imshow("eye", cv2.cvtColor(eye_img, cv2.COLOR_RGB2BGR))
        # cv2.waitKey(0)

        try:
            if eye_img.shape[0] == eye_img.shape[1]:
                cv2.imwrite(face_path, cv2.cvtColor(face_img, cv2.COLOR_RGB2BGR))
            # if(len(mouse_img)>0):
            if eye_img.shape[0] == eye_img.shape[1]:
                cv2.imwrite(mouse_path, cv2.cvtColor(mouse_img, cv2.COLOR_RGB2BGR))
            if eye_img.shape[0] == eye_img.shape[1]:
                cv2.imwrite(eye_path, cv2.cvtColor(eye_img, cv2.COLOR_RGB2BGR))
        except Exception as e:
            print(e)
            pass


def findAllFile(dir):
    fs = os.walk(dir)
    for list in fs:
        for f in list[2]:
            yield f


def face_detect(frame):
    detector = MTCNN()
    results = detector.detect_faces(frame)
    box = results[0]['box']
    ld = results[0]['keypoints']

    # face position
    phone_bia = (int)(0.7 * (max(box[2], box[3])))
    face_pos = ((max(0, ld['nose'][0]-phone_bia), max(0, ld['nose'][1]-phone_bia)),
                (min(frame.shape[1], ld['nose'][0]+phone_bia),
                 min(frame.shape[0], ld['nose'][1]+phone_bia)),)

    # eye position
    gapEx = (int)(0.3 * abs((ld['left_eye'][0] - ld['right_eye'][0])))
    gapEy = (int)(0.6 * (ld['nose'][1] - ld['left_eye'][1]))
    eye_bia = (int)(max(gapEx, gapEy))
    # print(eye_bia, ld['left_eye'])
    eye_pos = ((ld['left_eye'][0] - eye_bia, ld['left_eye'][1] - eye_bia),
                (ld['left_eye'][0] + eye_bia, ld['left_eye'][1] + eye_bia))

    #mouse position
    ctr_x = (int)(ld['mouth_left'][0] + 0.5 * abs((ld['mouth_left'][0] - ld['mouth_right'][0])))
    ctr_y = (int)(ld['mouth_left'][1] + 0.5 * abs((ld['mouth_left'][1] - ld['mouth_right'][1])))
    mouse_bia = (int)(0.75 * (ctr_y - ld['nose'][1]))
    mouse_pos = ((ctr_x - mouse_bia, ctr_y - mouse_bia),
                 (ctr_x + mouse_bia, ctr_y + mouse_bia))
    return face_pos, eye_pos, mouse_pos

def cut_pic(img_dir,lable):
    for f_jpg in findAllFile(img_dir):
        img_path = img_dir + f_jpg
        # print(img_path)
        img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        try:
            face_pos, eye_pos, mouse_pos = face_detect(img)
        except:
            pass
        # print(face_pos,eye_pos,mouse_pos)
        # print(str(face_pos[0][0])+ ':'+str(face_pos[1][0]), str(face_pos[0][1])+':'+str(face_pos[1][1]))
        # print(str(eye_pos[0][0]) + ':' + str(eye_pos[1][0]), str(eye_pos[0][1]) + ':' + str(eye_pos[1][1]))
        # print(str(mouse_pos[0][0]) + ':' + str(mouse_pos[1][0]), str(mouse_pos[0][1]) + ':' + str(mouse_pos[1][1]))
        save_dir = "./save_cut/" + "new_" + str(lable) + "/"
        # print(save_dir)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        cutter = CutPic(save_dir, img, face_pos, eye_pos, mouse_pos)
        cutter.cut()


if __name__ == "__main__":
    img_dir = "./save/"
    subdir = os.listdir(img_dir)
    for lable_dir in subdir:
        dir = img_dir + lable_dir + "/"
        print(dir + " begin.")
        cut_pic(dir,(int)(lable_dir))
        print(dir + " down.")
    # for f_jpg in findAllFile(img_dir):
    #     img_path = img_dir + f_jpg
    #     # print(img_path)
    #     img = Image.open(img_path)
    #     img = expand2square(img)
    #     img.show()
