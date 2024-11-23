import time

import cv2
import numpy as np
import math
import struct

def float_to_bytes(value):
    # 使用 '<f' 格式将浮点数转换为小端序字节
    return struct.pack('<f', value)

def detect_slope(frame):
    # 转换为灰度图像
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # 边缘检测
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    # 使用霍夫变换检测线条
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=80, minLineLength=160, maxLineGap=15)
    # 初始化变量
    slopes = []
    # 计算每条线的斜率并绘制
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if x2 - x1 != 0:  # 避免除以零
                slope = (y2 - y1) / (x2 - x1)
                angle = math.degrees(math.atan(slope))
                # 只考虑角度在±10度以内的线
                if abs(angle) <= 10:
                    slopes.append(slope)
                    cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    # 计算平均斜率
    average_slope = np.mean(slopes) if slopes else 0
    # 将斜率转换为角度
    angle = math.degrees(math.atan(average_slope))

    # 在图像中心绘制表示平均斜率的红线
    height, width = frame.shape[:2]
    center_x, center_y = width // 2, height // 2
    length = 1000  # 红线的长度
    dx = int(length / 2)
    dy = int(average_slope * dx)
    cv2.line(frame, (center_x - dx, center_y - dy), (center_x + dx, center_y + dy), (0, 0, 255), 2)

    return angle


if __name__ == '__main__':
    cap = cv2.VideoCapture(0)

    while True:
        ret = frame = 0
        while 1:
            ret, frame = cap.read()
            if ret:
                break
            time.sleep(0.01)

        angle = detect_slope(frame)
        print(f"Detected horizontal angle: {angle:.2f}°")

        cv2.imshow('Detected Lines', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()