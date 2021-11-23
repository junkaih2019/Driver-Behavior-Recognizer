import torch
from mtcnn_cv2 import MTCNN
from torch import nn
from torchvision import transforms

from Net.ResNet.ResNet50 import resnet50
from Net.mtcnn.FaceDetector import FaceDetector
import cv2
import time

view_frame_rate = True
view_behavior = True
view_box = True

# build network
mtcnn = MTCNN()
rsn50_call = resnet50(num_classes=2)
rsn50_smoke = resnet50(num_classes=2)
rsn50_mouse = resnet50(num_classes=2)
rsn50_eye = resnet50(num_classes=2)
rsn50_call.load_state_dict(torch.load('../Net/ResNet/saved_models/eye/eye_0.022_0.993_0.969.pkl'))
rsn50_smoke.load_state_dict(torch.load('../Net/ResNet/saved_models/eye/eye_0.019_0.994_0.936.pkl'))
rsn50_mouse.load_state_dict(torch.load('../Net/ResNet/saved_models/eye/eye_0.019_0.994_0.936.pkl'))
rsn50_eye.load_state_dict(torch.load('../Net/ResNet/saved_models/eye/eye_resnet50_0.pkl'))

# some modules to process the image
trans = [transforms.ToTensor()]
trans.insert(0, transforms.Resize(224))
trans = transforms.Compose(trans)
fcd = FaceDetector(mtcnn)
cap = cv2.VideoCapture(1)

def eval(net,img):
    net.eval()
    sfm = nn.Softmax(dim=1)
    p = sfm(net(img))
    # print(p)
    return '0' if p[0][0] > p[0][1] else '1'

while True:
    if view_frame_rate:
        start = time.time()
    ret, frame = cap.read()

    # nn
    try:
        # face mark points
        marks = mtcnn.detect_faces(frame)
        boxes = marks[0]['box']
        landmarks = marks[0]['keypoints']
        face_img, mouse_img, eye_img = fcd.get_3_pics(trans, frame, boxes, landmarks)
        #call_class = eval(rsn50_call,face_img)
        #smoke_class = eval(rsn50_call,mouse_img)
        #mouse_class = eval(rsn50_call,mouse_img)
        eye_class = eval(rsn50_call,eye_img)
        # behavior = call_class + smoke_class + mouse_class + eye_class

        # draw key boxes
        if view_box:
            probs = marks[0]['confidence']
            fcd.draw(frame, boxes, probs, landmarks)

        # show behavior
        if view_behavior:
            # cv2.putText(frame, "phone: " + call_class, (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255))
            # cv2.putText(frame, "mouse: " + mouse_class, (30, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255))
            cv2.putText(frame, "eye: " + eye_class, (30, 160), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255))
            # cv2.putText(frame, "smoke: " + smoke_class, (30, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255))
    except Exception as e:
        print(e)
        pass



    # show frame rate
    if view_frame_rate:
        end = time.time()
        seconds = end - start
        fps = 1 / seconds
        cv2.putText(frame, "fps: " + str(round(fps)), (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255))

    cv2.imshow('face detector', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()