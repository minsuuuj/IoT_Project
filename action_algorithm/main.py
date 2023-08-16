import json
import trt_pose.coco
import trt_pose.models
import torch
import torch2trt
from torch2trt import TRTModule
import time, sys
import cv2
import torchvision.transforms as transforms
import PIL.Image
from trt_pose.draw_objects import DrawObjects
from trt_pose.parse_objects import ParseObjects
import argparse
import os.path
import json
import numpy as np
from action import *

'''
hnum: 0 based human index
kpoint : keypoints (float type range : 0.0 ~ 1.0 ==> later multiply by image width, height
'''

def overlay_action(img, action_list):
    action_key = ['standing', 'sitting', 'lying', 'flip', 'eating', 'falling']
    for i, action in enumerate(action_list):
        if action:
            cv2.putText(img, action_key[i], (10,20+i*10), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 1)

def get_keypoint(humans, hnum, peaks):
    #check invalid human index
    kpoint = []
    human = humans[0][hnum]
    C = human.shape[0]
    for j in range(C):
        k = int(human[j])
        if k >= 0:
            peak = peaks[0][j][k]   # peak[1]:width, peak[0]:height
            peak = (j, float(peak[0]), float(peak[1]))
            kpoint.append(peak)
            #print('index:%d : success [%5.3f, %5.3f]'%(j, peak[1], peak[2]) )
        else:    
            peak = (j, None, None)
            kpoint.append(peak)
            #print('index:%d : None %d'%(j, k) )
    return kpoint

parser = argparse.ArgumentParser(description='TensorRT pose estimation run')
parser.add_argument('--model', type=str, default='densenet', help = 'resnet or densenet' )
args = parser.parse_args()

with open('human_pose.json', 'r') as f:
    human_pose = json.load(f)

topology = trt_pose.coco.coco_category_to_topology(human_pose)

num_parts = len(human_pose['keypoints'])
num_links = len(human_pose['skeleton'])

if 'resnet' in args.model:
    print('------ model = resnet--------')
    MODEL_WEIGHTS = './pth_files/resnet18_baseline_att_224x224_A_epoch_249.pth'
    OPTIMIZED_MODEL = './pth_files/resnet18_baseline_att_224x224_A_epoch_249_trt.pth'
    model = trt_pose.models.resnet18_baseline_att(num_parts, 2 * num_links).cuda().eval()
    WIDTH = 224
    HEIGHT = 224

else:    
    print('------ model = densenet--------')
    MODEL_WEIGHTS = './pth_files/densenet121_baseline_att_256x256_B_epoch_160.pth'
    OPTIMIZED_MODEL = './pth_files/densenet121_baseline_att_256x256_B_epoch_160_trt.pth'
    model = trt_pose.models.densenet121_baseline_att(num_parts, 2 * num_links).cuda().eval()
    WIDTH = 256
    HEIGHT = 256

data = torch.zeros((1, 3, HEIGHT, WIDTH)).cuda()
if os.path.exists(OPTIMIZED_MODEL) == False:
    model.load_state_dict(torch.load(MODEL_WEIGHTS))
    model_trt = torch2trt.torch2trt(model, [data], fp16_mode=True, max_workspace_size=1<<25)
    torch.save(model_trt.state_dict(), OPTIMIZED_MODEL)

model_trt = TRTModule()
model_trt.load_state_dict(torch.load(OPTIMIZED_MODEL))

t0 = time.time()
torch.cuda.current_stream().synchronize()
for i in range(50):
    y = model_trt(data)
torch.cuda.current_stream().synchronize()
t1 = time.time()

print(50.0 / (t1 - t0))

mean = torch.Tensor([0.485, 0.456, 0.406]).cuda()
std = torch.Tensor([0.229, 0.224, 0.225]).cuda()
device = torch.device('cuda')

def preprocess(image):
    global device
    device = torch.device('cuda')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = PIL.Image.fromarray(image)
    image = transforms.functional.to_tensor(image).to(device)
    image.sub_(mean[:, None, None]).div_(std[:, None, None])
    return image[None, ...]

def draw_bounding_box(img, src, keypoints):
    bottom_coor = (round(keypoints.bottom[1] * WIDTH * X_compress), round(keypoints.bottom[0] * HEIGHT * Y_compress))
    top_coor = (round(keypoints.top[1] * WIDTH * X_compress), round(keypoints.top[0] * HEIGHT * Y_compress))
    cv2.rectangle(src, bottom_coor, top_coor, (255, 100, 0, 100), 2)

    # transparent_color = (0, 0, 255, 100)  # (B, G, R, Alpha)

    # # 색칠할 영역 생성
    # overlay = img.copy()
    # cv2.rectangle(overlay, (100, 100), (200, 200), transparent_color, -1)

    # # 영역을 원본 이미지에 합성
    # cv2.addWeighted(overlay, 0.5, src, 0.5, 0, src)

def execute(img, src, t, frame_cnt):
    color = (255, 0, 0)
    data = preprocess(img)
    cmap, paf = model_trt(data)
    cmap, paf = cmap.detach().cpu(), paf.detach().cpu()
    counts, objects, peaks = parse_objects(cmap, paf)#, cmap_threshold=0.15, link_threshold=0.15)
    fps = 1.0 / (time.time() - t)

    for i in range(counts[0]):
        keypoints = Kpoints(get_keypoint(objects, i, peaks))

        mc_x = round(keypoints.mc[1] * WIDTH * X_compress)
        mc_y = round(keypoints.mc[0] * HEIGHT * Y_compress)
        cv2.circle(src, (mc_x, mc_y), 5, (255, 255, 255), 7)

        if keypoints.check_box_point():
            draw_bounding_box(img, src, keypoints)
        
        if frame_cnt == 0:
            action_recog.append(keypoints)
            action_recog()    # onehot encoding [서기, 앉기, 눕기, 뒤집어 자기, 먹기, 낙하]
            action_recog.print_action()
        overlay_action(src, action_recog.action_status)

    # print("FPS:%f "%(fps))
    draw_objects(src, counts, objects, peaks)

    # cv2.putText(src, "FPS: %f" % (fps), (20, 20),  cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1)
    out_video.write(src)

    return src

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

ret_val, img = cap.read()
fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
out_video = cv2.VideoWriter('test.mp4', fourcc, cap.get(cv2.CAP_PROP_FPS), (640, 480))
count = 0

X_compress = 640.0 / WIDTH * 1.0
Y_compress = 480.0 / HEIGHT * 1.0

if cap is None:
    print("Camera Open Error")
    sys.exit(0)

parse_objects = ParseObjects(topology)
draw_objects = DrawObjects(topology)
action_recog = Action_recognition()

frame_num = 0

while cap.isOpened() and True:
    if frame_num == 30:
        frame_num = 0
    t = time.time()
    ret_val, dst = cap.read()
    if ret_val == False:
        print("Camera read Error")
        break
    
    img = cv2.resize(dst, dsize=(WIDTH, HEIGHT), interpolation=cv2.INTER_AREA)
    src = execute(img, dst, t, frame_num)

    cv2.imshow('src', src)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    count += 1
    frame_num += 1

cv2.destroyAllWindows()
out_video.release()
cap.release()