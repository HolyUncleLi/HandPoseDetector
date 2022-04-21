import cv2
from tqdm import tqdm
import mediapipe as mp
import csv
import time

mpHands = mp.solutions.hands
# Hands是一个类，有四个初始化参数，static_image_mode,max_num_hands,min_detection_confidence,min_tracking_confidence
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils  # 画线函数


root = "E:\\sample\\hands\\"
video_name = "0"
f_train = open(root+"origin\\case\\"+video_name+".csv",'w',newline='')
csv_write = csv.writer(f_train,dialect='excel')

# 视频处理函数
def generate_video(root_path, name):
    #  打开单个样本实例的csv文件
    f_train = open(root + "train\\case\\" + name + ".csv", 'w', newline='')
    csv_write = csv.writer(f_train, dialect='excel')

    print('视频开始处理')
    input_path = root_path + name + '.mp4'  # 视频数据路径
    cap = cv2.VideoCapture( input_path )
    frame_count = 0
    while (cap.isOpened()):
        success, frame = cap.read()
        frame_count += 1
        if not success:
            break
    cap.release()
    print('视频总帧数为', frame_count)

    cap = cv2.VideoCapture(input_path)
    frame_size = (cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = cap.get(cv2.CAP_PROP_FPS)

    with tqdm(total=frame_count - 1) as pbar:
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
                                temp_1.append(lm.x)  # x方向坐标
                                temp_1.append(lm.y)  # y方向坐标
                                temp_1.append(lm.z)  # z方向坐标
                        csv_write.writerow(temp_1)  # 共63个数据
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


def main():
    generate_video('E:/sample/hands/origin/case/',video_name)


if __name__ == "__main__":
    main()
