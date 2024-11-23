import time
import serial
import numpy as np
import cv2


import qt
from PyQt6.QtCore import pyqtSignal, QObject
import threading


class Stm32(QObject):
    update_qr_signal = pyqtSignal()
    update_weight_signal = pyqtSignal()
    update_cross_signal =  pyqtSignal()
    update_serial_signal = pyqtSignal(str)  # 新增serial错误信号
    update_line_signal = pyqtSignal()

    def __init__(self, ser_32):
        super(Stm32, self).__init__()
        self.ser_32_path = ser_32
        self.ser = None

        self.cmd_type = ["0", "QR_ack", "QR_recv",
                         "Cross_ack", "Cross_recv",
                         "Weight_ack", "Weight_recv",
                         "clock_ack", "clock_recv",
                         "line_ack", "line_recv",]
        self.scan_th = threading.Thread(target=self._recv_data, daemon=True)

    def send_data(self, cmd_type, data):
        index = self.cmd_type.index(cmd_type)
        cmd = [len(data) + 2, index]
        cmd += data
        self.ser.write(bytes(cmd))

    def qr_update(self):
        self.update_qr_signal.emit()

    def cross_update(self):
        self.update_cross_signal.emit()

    def weight_update(self):
        self.update_weight_signal.emit()

    def serial_update(self, state):
        self.update_serial_signal.emit(state)

    def line_update(self):
        self.update_line_signal.emit()

    def start_run(self):
        self.scan_th.start()

    def _recv_data(self):
        recv_num = -1   # -1代表正在等待接收
        while True:
            try:
                self.ser = serial.Serial(self.ser_32_path, baudrate=115200, timeout=1)
                self.ser.read(self.ser.in_waiting)
                while True:
                    self.serial_update(f"connected stm32")
                    if self.ser.in_waiting != 0:
                        if recv_num == -1:
                            recv_num = int(self.ser.read(1)[0])
                        elif self.ser.in_waiting >= recv_num - 1:   # 读取到所有数据
                            act = int(self.ser.read(1)[0])
                            print(f"act={act}")
                            print(self.cmd_type[act])

                            if self.cmd_type[act] == "QR_ack":
                                self.qr_update()
                            elif self.cmd_type[act] == "Cross_ack":
                                self.cross_update()
                            elif self.cmd_type[act] == "Weight_ack":
                                self.weight_update()
                            elif self.cmd_type[act] == "clock_ack":
                                self.send_data("clock_recv", [])

                            elif self.cmd_type[act] == "line_ack":
                                self.line_update()
                            else:
                                self.ser.read(self.ser.in_waiting)
                            recv_num = -1      # -1代表正在等待接收

                    elif qt.button_activate != "":
                        if qt.button_activate == "QR_ack":
                            self.qr_update()
                        if qt.button_activate == "Cross_ack":
                            self.cross_update()
                        if qt.button_activate == "Weight_ack":
                            self.weight_update()
                        if qt.button_activate == "Line_ack":
                            self.line_update()
                        qt.button_activate = ""
                    elif qt.ser_sent == 1:
                        qt.ser_sent = 0
                        self.ser.write(qt.ser_out)
                        try:
                            print(f"ser write: \"{qt.ser_out.decode('utf-8')}\"")
                        except:
                            print("ser write a unread code!")
                    else:
                        time.sleep(0.01)

            except OSError as e:
                self.serial_update("serial error!!")
                time.sleep(0.3)  # 等待0.3秒后重试

