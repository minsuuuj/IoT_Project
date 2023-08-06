import json
import numpy as np

keypoints_name = ['nose',
                'eye_left', 'eye_right',
                'ear_left', 'ear_right',
                'shoulder_left', 'shoulder_right',
                'elbow_left', 'elbow_right',
                'wrist_left', 'wrist_right',
                'hip_left', 'hip_right',
                'knee_left', 'knee_right',
                'ankle_left', 'ankle_right',
                'neck']

class Kpoints:
    def __init__(self, keypoint_list):
        self.nose = keypoint_list[0]
        self.shoulder_left = keypoint_list[1]
        self.shoulder_right = keypoint_list[2]
        self.elbow_left = keypoint_list[3]
        self.elbow_right = keypoint_list[4]
        self.wrist_left = keypoint_list[5]
        self.wrist_right = keypoint_list[6]
        self.hip_left = keypoint_list[7]
        self.hip_right = keypoint_list[8]
        self.knee_left = keypoint_list[9]
        self.knee_right = keypoint_list[10]
        self.ankle_left = keypoint_list[11]
        self.ankle_right = keypoint_list[12]
        self.neck = keypoint_list[13]

        # mass center 구하기
        total_x = 0
        total_y = 0
        for x, y in keypoint_list:
            total_x =+ x
            total_y =+ y
        self.mc_x = total_x / len(keypoint_list)
        self.mc_y = total_y / len(keypoint_list)
    
    def __iter__(self):
        fields = [self.nose, self.shoulder_left, self.shoulder_right, self.elbow_left, self.elbow_right, self.wrist_left, self.wrist_right, self.hip_left, self.hip_right, self.knee_left, self.knee_right, self.ankle_left, self.ankle_right, self.neck,]

        for field in fields:
            yield field    

frame = '1-1_606-C08_2D.json'

def nasNF_to_trt_calibration(nas_keypoints):
    nas_keypoints_fair = []
    for i in range(14):
        j = i*3
        fair = (nas_keypoints[j], nas_keypoints[j+1])
        nas_keypoints_fair.append(fair)
    sorted_keypoints = []

    order_list = [7, 11, 10, 12, 9, 13, 8, 3, 2, 4, 1, 5, 0, 6]
    
    for i in order_list:
        sorted_keypoints.append(nas_keypoints_fair[i])

    return sorted_keypoints
    

with open(frame, 'r') as data:
    C08_606 = json.load(data)
    frame_keypoints = C08_606['annotations'][0]['keypoints']
    frame_keypoints = nasNF_to_trt_calibration(frame_keypoints)
    curr = Kpoints(frame_keypoints)

for keypoint in curr:
    print(keypoint)


width_sholder = abs(curr.shoulder_left[0] - curr.shoulder_right[0])
height = abs(curr.neck[1] - (curr.ankle_left[1] + curr.ankle_right[1]) / 2)

print(width_sholder, height)
print(curr.mc_x, curr.mc_y)
