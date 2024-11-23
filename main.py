import qt
import yolo
import threading
import stm32
import sys
import os


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(sys.argv[0]))

    cross_path = os.path.join(base_dir, "lib/ring3.onnx")
    weight_path = os.path.join(base_dir, "lib/weight3.onnx")
    ui_path = os.path.join(base_dir, "lib/main.ui")
    log_path = os.path.join(base_dir, "lib/run.log")
    # sys.stdout = open(log_path, 'w')

    print("start run")



    mode_cross = yolo.OnnxFrame(cross_path, "blue", "red", "green")
    mode_weight = yolo.OnnxFrame(weight_path, "red", "green", "blue")
    cap = yolo.VideoCapture(0)
    mode_cross.infer_img(cap.read()[1])
    mode_weight.infer_img(cap.read()[1])


    window = qt.MyWindow(mode_cross, mode_weight, cap, ui_path, "/dev/usb2d")

    window.run()



