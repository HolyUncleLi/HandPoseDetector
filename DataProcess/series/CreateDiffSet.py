import cv2
from tqdm import tqdm
import mediapipe as mp
import csv
import time

mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils


root = "E:\\sample\\hands\\"
num_train = 0
f_train = open(root+"train\\grasp\\"+str(num_train)+".csv",'w',newline='')
csv_write = csv.writer(f_train,dialect='excel')

# 视频处理函数
def generate_video(root_path, sum):


    # 依次处理所有视频数据
    for i in range(sum):
        last_frame = []
        for j in range(64):
            last_frame.append(float(0))
        #  打开单个样本实例的csv文件
        f_train = open(root + "train\\grasp\\" + str(i) + ".csv", 'w', newline='')
        csv_write = csv.writer(f_train, dialect='excel')


        print('视频开始处理')
        input_path = root_path + str(i) + '.mp4'  # 视频数据路径
        cap = cv2.VideoCapture(input_path)
        frame_count = 0
        while(cap.isOpened()):
            success,frame = cap.read()
            frame_count += 1
            if not success:
                break
        cap.release()
        print('视频总帧数为',frame_count)

        cap = cv2.VideoCapture(input_path)
        frame_size = (cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = cap.get(cv2.CAP_PROP_FPS)


        with tqdm(total=frame_count-1) as pbar:
            try:
                count = 1  # 记录帧数
                while cap.isOpened():
                    success, frame = cap.read()
                    if not success:
                        break
                    try:
                        # 对每一帧进行处理
                        # 预测单张帧图像中的手势坐标
                        results = hands.process(frame)
                        if results.multi_hand_landmarks:  # 检测到手势
                            for handLms in results.multi_hand_world_landmarks:
                                # landmark有21个，id是索引，lm是x,y,z坐标
                                temp_1 = [count]
                                for id, lm in enumerate(handLms.landmark):
                                    # 将landmark的比例坐标转换为在图像像元上的坐标
                                    # cx, cy, cz = lm.x, lm.y, lm.z
                                    temp_1.append(lm.x * 10)  # x方向坐标
                                    temp_1.append(lm.y * 10)  # y方向坐标
                                    temp_1.append(lm.z * 10)  # z方向坐标

                            for k in range(1,64,1):
                                temp = temp_1[k]
                                if abs(temp_1[k] - last_frame[k]) >= 0.0001:  # 忽略过小值
                                    temp_1[k] = temp_1[k] - last_frame[k]      # 计算差值
                                else:
                                    temp_1[k] = float(0)
                                last_frame[k] = temp

                            csv_write.writerow(temp_1)  # 共63个数据写入本地
                            count += 1  # 处理帧数自增1
                    except:
                        print('error')
                        pass
                    if success == True:
                        pbar.update(1)
            except:
                print('error2')
                pass
        cv2.destroyAllWindows()
        cap.release()
        f_train.close()

def main():
    generate_video('E:/sample/hands/origin/grasp/', 30)


if __name__ == "__main__":
    main()
