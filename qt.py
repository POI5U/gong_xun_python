import threading
from idlelib.iomenu import encoding

from PyQt6.QtWidgets import QLabel, QLineEdit, QTextEdit, QComboBox, QPushButton
from PyQt6.QtWidgets import QApplication, QWidget
from PyQt6.QtCore import Qt, QEvent
from PyQt6 import uic
import sys
import cv2
from PyQt6.QtGui import QImage, QPixmap
import time

import line
import stm32
import numpy as np
from PyQt6.QtCore import pyqtSignal

button_activate = ""
ser_out = bytes([])
ser_sent = 0

class MyWindow:
    def __init__(self, mode_cross, mode_weight, cap, ui_path, ser_32):
        self.app = QApplication(sys.argv)
        self.my_ui: QWidget = uic.loadUi(ui_path)
        self.state = "当前: 调试模式"
        self.move_num = ""          # 实际使用的顺序编号
        self.QR_Enable = True
        self.cap = cap
        self.mode_cross = mode_cross
        self.mode_weight = mode_weight

        # 实列化32, 链接qr和cross信号
        self.worker = stm32.Stm32(ser_32)
        self.worker.update_qr_signal.connect(self.qr_update)
        self.worker.update_cross_signal.connect(self.cross_update)
        self.worker.update_weight_signal.connect(self.weight_update)
        self.worker.update_serial_signal.connect(self.state_serial)
        self.worker.update_line_signal.connect(self.line_update)

        # 加载控件: 关闭按钮
        self.pushButton_Esc: QPushButton = self.my_ui.pushButton_Esc
        self.pushButton_Esc.clicked.connect(self.__button_esc)
        # 加载控件: 切换模式按钮
        self.pushButton_Change: QPushButton = self.my_ui.pushButton_Change
        self.pushButton_Change.clicked.connect(self.__button_change)
        # 加载控件: 运行按钮
        self.pushButton_Start: QPushButton = self.my_ui.pushButton_Start
        self.pushButton_Start.clicked.connect(self.__button_start)
        # 加载控件: 文字显示模块
        self.textEdit: QTextEdit = self.my_ui.textEdit
        self.textEdit.textChanged.connect(self.handle_text_changed)
        # 加载图像显示框label
        self.label_1: QLabel = self.my_ui.label_1
        self.label_2: QLabel = self.my_ui.label_2
        self.label_3: QLabel = self.my_ui.label_3
        # 加载控件: qr启动按钮
        self.pushButton_QR: QPushButton = self.my_ui.pushButton_QR
        self.pushButton_QR.clicked.connect(self.__button_qr)
        # 加载控件: ring启动按钮
        self.pushButton_Ring: QPushButton = self.my_ui.pushButton_Ring
        self.pushButton_Ring.clicked.connect(self.__button_ring)
        # 加载控件: weight启动按钮
        self.pushButton_Weight: QPushButton = self.my_ui.pushButton_Weight
        self.pushButton_Weight.clicked.connect(self.__button_weight)
        # 加载控件: line启动按钮
        self.pushButton_Line: QPushButton = self.my_ui.pushButton_Line
        self.pushButton_Line.clicked.connect(self.__button_line)
        # 加载控件: 串口信息显示框
        self.textEdit_serial: QTextEdit = self.my_ui.textEdit_serial
        self.textEdit.setFocus()

    def show_frame(self, lab_num, rgb_img):
        if lab_num == 2:
            show_lab = self.label_2
        elif lab_num == 3:
            show_lab = self.label_3
        else:
            show_lab = self.label_1

        h, w, ch = rgb_img.shape
        window_w = int(show_lab.geometry().width())
        window_h = int(show_lab.geometry().height())
        if window_w != w or window_h != h:
            rgb_img = cv2.resize(rgb_img, (window_w, window_h))
        qt_image = QImage(rgb_img.data, window_w, window_h, QImage.Format.Format_RGB888)
        qt_pixmap = QPixmap.fromImage(qt_image)
        show_lab.setPixmap(
            qt_pixmap.scaled(400, 300, Qt.AspectRatioMode.KeepAspectRatio))

    def qr_update(self):
        self.textEdit.setFocus()
        self.QR_Enable = True

    def cross_update(self):
        global ser_out, ser_sent
        img, yes, strs = self.mode_scan(self.mode_cross)
        if yes:
            ser_out = bytes(f"{chr(13)}{chr(4)}" + strs[:-1], encoding='utf8')
        else:
            ser_out = bytes(f"{chr(2)}{chr(4)}", encoding='utf8')
        ser_sent = 1
        self.show_frame(1, cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    def weight_update(self):
        global ser_out, ser_sent
        img, yes, strs = self.mode_scan(self.mode_weight)
        if yes:
            ser_out = bytes(f"{chr(14)}{chr(6)}" + strs, encoding='utf8')
        else:
            ser_out = bytes(f"{chr(2)}{chr(6)}", encoding='utf8')
        ser_sent = 1
        self.show_frame(2, cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    def line_update(self):
        global ser_out, ser_sent
        img0 = self.cap.read()[1]
        angle = line.detect_slope(img0)
        print(f"line angle == {angle}")
        send_bytes = line.float_to_bytes(angle)

        ser_out = bytes(f"{chr(6)}{chr(10)}", encoding='utf8') + send_bytes
        ser_sent = 1
        self.show_frame(3, cv2.cvtColor(img0, cv2.COLOR_BGR2RGB))

    def mode_scan(self, mode):
        img0 = self.cap.read()[1]
        det_boxes, scores, ids = mode.infer_img(img0, 0.4, 0.5)
        if len(det_boxes) > 0:
            # 结果绘图
            for box, score, id1 in zip(det_boxes, scores, ids):
                label = '%s:%.2f' % (mode.dic_labels[id1], score)
                box = box.astype(np.int32)
                mode.plot_one_box(box.astype(np.int32), img0,
                                       (0, 255, 0), in_label=label, line_thickness=1)
            # 遍历数组以得到绿色的作为best_idx
            green_indices = [i for i, id1 in enumerate(ids) if mode.dic_labels[id1] == 'green']
            if green_indices:
                best_idx = green_indices[0]  # 选择第一个绿色物体
                best_box = det_boxes[best_idx]
            else:   # 没找到绿色的则选择置信度最高的
                best_idx = np.argmax(scores)
                best_box = det_boxes[best_idx]

            # 计算中心坐标
            x_ = int((best_box[0] + best_box[2]) / 2)
            y_ = int((best_box[1] + best_box[3]) / 2)
            colors = ['r', 'g', 'b']
            sent=f"x={str(x_).zfill(3)},y={str(y_).zfill(3)}{colors[ids[best_idx]]}"
            return img0, True, sent
        else:
            return img0, False, f""

    def handle_text_changed(self):
        global ser_out, ser_sent
        all_text = self.textEdit.toPlainText()      # 获取QTextEdit中的所有文本
        if '\n' in all_text:        # 检查是否包含Enter键
            lines = all_text.split('\n')
            last_line = lines[-2] if len(lines) > 1 else lines[-1]
            # 设置QTextEdit的文本
            self.textEdit.blockSignals(True)  # 暂时阻止信号，以避免递归调用
            if self.QR_Enable:  # 读取成功
                self.textEdit.setPlainText(last_line)
                self.move_num = last_line

                self.QR_Enable = False
                send_str = f"{chr(9)}{chr(2)}" + last_line
                ser_out = bytes(send_str, encoding='utf8')
                ser_sent = 1
            else:
                self.textEdit.setPlainText(self.move_num)
            self.textEdit.blockSignals(False)  # 恢复信号

    def run(self):
        self.worker.start_run()
        self.my_ui.show()               # 显示窗口
        #self.my_ui.showFullScreen()     # 全屏
        self.app.exec()       # 启动

    def __button_esc(self):
        sys.exit()

    def __button_start(self):
        self.textEdit.setFocus()

    def __button_change(self):
        if self.state == "当前: 调试模式":
            self.state = "当前: 运行模式"
            self.pushButton_Change.setText(self.state)
        else:
            self.state = "当前: 调试模式"
            self.pushButton_Change.setText(self.state)

    def __button_qr(self):
        global button_activate
        button_activate = "QR_ack"

    def __button_ring(self):
        global button_activate
        button_activate = "Cross_ack"

    def __button_weight(self):
        global button_activate
        button_activate = "Weight_ack"

    def __button_line(self):
        global button_activate
        button_activate = "Line_ack"

    def state_serial(self, state):
        self.textEdit_serial.setText(state)

