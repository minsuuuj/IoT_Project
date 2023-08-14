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

def calc_dist(vector1, vector2):
    if vector1[1] and vector2[1]:
        distance = np.sum((np.array(vector1) - np.array(vector2))**2)**0.5
        return distance
    else:
        return None
        
def calc_center(vector1, vector2):
    return list((np.array(vector1)+np.array(vector2))/2)

class Kpoints:
    # 좌표는 (y, x) fair
    def __init__(self, keypoint_list):
        self.num_keypoints = len(keypoint_list)

        self.nose = keypoint_list[0][1:]
        self.eye_left = keypoint_list[1][1:]
        self.eye_right = keypoint_list[2][1:]
        self.ear_left = keypoint_list[3][1:]
        self.ear_right = keypoint_list[4][1:]
        self.shoulder_left = keypoint_list[5][1:]
        self.shoulder_right = keypoint_list[6][1:]
        self.elbow_left = keypoint_list[7][1:]
        self.elbow_right = keypoint_list[8][1:]
        self.wrist_left = keypoint_list[9][1:]
        self.wrist_right = keypoint_list[10][1:]
        self.hip_left = keypoint_list[11][1:]
        self.hip_right = keypoint_list[12][1:]
        self.knee_left = keypoint_list[13][1:]
        self.knee_right = keypoint_list[14][1:]
        self.ankle_left = keypoint_list[15][1:]
        self.ankle_right = keypoint_list[16][1:]
        self.neck = keypoint_list[17][1:]

        self.mc = self.calc_mc(keypoint_list)   # [mc_y, mc_x]

        self.is_box_point = self.check_box_point()

        if self.is_box_point:
            self.bottom, self.top = self.find_bot_top()
        else:
            self.bottom, self.top = [None, None], [None, None]
    
    def __iter__(self):
        fields = [self.nose, self.shoulder_left, self.shoulder_right, self.elbow_left, self.elbow_right, self.wrist_left, self.wrist_right,\
                  self.hip_left, self.hip_right, self.knee_left, self.knee_right, self.ankle_left, self.ankle_right, self.neck,]

        for field in fields:
            yield field

    def calc_mc(self, keypoints):
        keypoints_x = np.array(np.array(keypoints)[:,2])
        keypoints_x = keypoints_x[keypoints_x != None]
        keypoints_y = np.array(np.array(keypoints)[:,1])
        keypoints_y = keypoints_y[keypoints_y != None]

        try:
            mc_x = np.mean(keypoints_x)
            mc_y = np.mean(keypoints_y)
        except:
            mc_x = 0
            mc_y = 0

        return [mc_y, mc_x]
    
    def check_box_point(self):
        box_point_list = [self.shoulder_left, self.shoulder_right, self.ankle_left, self.ankle_right]
        for p in box_point_list:
            if None in p:
                return False
        return True 
    
    def find_bot_top(self):
        box_point_list = [self.shoulder_left, self.shoulder_right, self.ankle_left, self.ankle_right]
        box_point_list = [box_point for box_point in box_point_list if box_point[0] != None]

        bottom_x = 0
        bottom_y = 0
        top_x = 1
        top_y = 1
        for p in box_point_list:
            x, y = p
            if x - bottom_x > 0:
                bottom_x = x
            if y - bottom_y > 0:
                bottom_y = y

            if x - top_x < 0:
                top_x = x
            if y - top_y < 0:
                top_y = y

        return [bottom_x, bottom_y], [top_x, top_y]
    

class Keypoint_sequence:
    def __init__(self):
        self.before2: Kpoints = None
        self.before1: Kpoints = None
        self.present: Kpoints = None

    def append(self, kp):
        self.present = kp
        self.before1 = self.present
        self.before2 = self.before1
        
        
class Action_recognition(Keypoint_sequence):
    falling_threshold = 0.5
    standing_threshold = 4
    sitting_threshold = 2
    lying_threshold = 1.5
    eating_threshold = 0.3

    # onehot encoding [서기, 앉기, 눕기, 뒤집어 자기, 먹기, 떨어지기]
    action_status = [False] * 6

    def __init__(self):
        self.bb_ratio: float = None

    def __call__(self):
        self.action_status = [False] * 6
        if self.present.is_box_point:
            bb_width = self.present.bottom[1] - self.present.top[1]
            bb_height = self.present.bottom[0] - self.present.top[0]
            self.bb_ratio = bb_height / bb_width

            self.is_standing()
            self.is_sitting()
            self.is_lying()
        else:
            self.bb_ratio = None
            
        self.is_eating()
        self.is_falling_down()

        return self.action_status
        
    def is_standing(self):
        if self.bb_ratio - self.standing_threshold > 0:
            self.action_status[0] = True

    def is_sitting(self):
        if not self.action_status[0]:
            if self.bb_ratio - self.sitting_threshold > 0:
                self.action_status[1] = True

    def is_lying(self):
        if not self.action_status[0] and not self.action_status[1]:
            if self.bb_ratio < self.lying_threshold:
                self.action_status[2] = True
                if self.present.nose[0] is None:
                    self.action_status[3] = True

    def is_eating(self):
        try:
            if calc_dist(self.present.wrist_left, self.present.nose) < self.eating_threshold \
                or calc_dist(self.present.wrist_right, self.present.nose) < self.eating_threshold:
                self.action_status[4] = True
        except:
            pass
    
    def is_falling_down(self):
        try:
            if self.before1:
                moving_dist_mc = calc_dist(self.before1.mc, self.present.mc)
                if moving_dist_mc > self.falling_threshold:
                    self.action_status[5] = True
        except:
            pass

    def print_action(self, src):
        action_key = ['서기', '앉기', '눕기', '뒤집어 자기', '먹기', '낙하']

        for i, check in enumerate(self.action_status):
            if check:
                print(action_key[i], end=' ')
        print()