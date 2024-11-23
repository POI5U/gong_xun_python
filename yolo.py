import cv2
import numpy as np
from numpy import random
import threading
import queue
import onnxruntime as ort
import time


class VideoCapture:
    # 自定义无缓存读视频类
    def __init__(self, camera_id):
        self.cap = cv2.VideoCapture(camera_id)
        self.q = queue.Queue(maxsize=3)
        self.stop_threads = False  # 关闭读取线程
        th = threading.Thread(target=self._reader)
        th.daemon = True  # 设置工作线程为后台运行
        th.start()

    def _reader(self):
        # 实时读帧，只保存最后一帧
        while not self.stop_threads:
            ret, frame = self.cap.read()
            if not ret:
                break
            if not self.q.empty():
                try:
                    self.q.get_nowait()
                except queue.Empty:
                    pass
            self.q.put(frame)

    def read(self):
        return 1, self.q.get()

    def release(self):
        self.stop_threads = True
        self.cap.release()


class OnnxFrame:
    def __init__(self, onnx_model_path, *args):
        # 模型加载
        so = ort.SessionOptions()
        self.net = ort.InferenceSession(onnx_model_path, so)
        # 标签字典
        # self.dic_labels = {0: 'blue', 1: 'red', 2: 'green'}
        self.dic_labels = {i: label for i, label in enumerate(args)}
        # 模型参数
        self.model_h = 320
        self.model_w = 320
        self.nl = 3
        self.na = 3
        self.stride = [8., 16., 32.]
        self.anchors = [[10, 13, 16, 30, 33, 23], [30, 61, 62, 45, 59, 119], [116, 90, 156, 198, 373, 326]]
        self.anchor_grid = np.asarray(self.anchors, dtype=np.float32).reshape(self.nl, -1, 2)

        self.center_x = 0
        self.center_y = 0

    def plot_one_box(self, x, img, color=None, in_label=None, line_thickness=None):
        x = x.squeeze()
        tl = (line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1)
        color = color or [random.randint(0, 255) for _ in range(3)]
        c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))

        cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
        # 计算矩形中心点
        self.center_x = (c1[0] + c2[0]) // 2
        self.center_y = (c1[1] + c2[1]) // 2
        # 在矩形中心画一个小圆点
        cv2.circle(img, (self.center_x, self.center_y),
                   radius=5, color=color, thickness=-1, lineType=cv2.LINE_AA)
        if in_label:
            tf = max(tl - 1, 1)  # font thickness
            t_size = cv2.getTextSize(in_label, 0, fontScale=tl / 3, thickness=tf)[0]
            c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
            cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
            cv2.putText(img, in_label, (c1[0], c1[1] - 2), 0,
                        tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)

    @staticmethod
    def _make_grid(nx, ny):
        xv, yv = np.meshgrid(np.arange(ny), np.arange(nx))
        return np.stack((xv, yv), 2).reshape((-1, 2)).astype(np.float32)

    def cal_outputs(self, outs):
        row_ind = 0
        grid = [np.zeros(1)] * self.nl
        for i in range(self.nl):
            h, w = int(self.model_w / self.stride[i]), int(self.model_h / self.stride[i])
            length = int(self.na * h * w)
            if grid[i].shape[2:4] != (h, w):
                grid[i] = self._make_grid(w, h)

            outs[row_ind:row_ind + length, 0:2] = (outs[row_ind:row_ind + length, 0:2] * 2. - 0.5 + np.tile(
                grid[i], (self.na, 1))) * int(self.stride[i])
            outs[row_ind:row_ind + length, 2:4] = (outs[row_ind:row_ind + length, 2:4] * 2) ** 2 * np.repeat(
                self.anchor_grid[i], h * w, axis=0)
            row_ind += length
        return outs

    def post_process_opencv(self, outputs, img_h, img_w, thread_nms, thread_cond):
        conf = outputs[:, 4].tolist()
        c_x = outputs[:, 0] / self.model_w * img_w
        c_y = outputs[:, 1] / self.model_h * img_h
        w = outputs[:, 2] / self.model_w * img_w
        h = outputs[:, 3] / self.model_h * img_h
        p_cls = outputs[:, 5:]
        if len(p_cls.shape) == 1:
            p_cls = np.expand_dims(p_cls, 1)
        cls_id = np.argmax(p_cls, axis=1)

        p_x1 = np.expand_dims(c_x - w / 2, -1)
        p_y1 = np.expand_dims(c_y - h / 2, -1)
        p_x2 = np.expand_dims(c_x + w / 2, -1)
        p_y2 = np.expand_dims(c_y + h / 2, -1)
        areas = np.concatenate((p_x1, p_y1, p_x2, p_y2), axis=-1)

        areas = areas.tolist()
        in_ids = cv2.dnn.NMSBoxes(areas, conf, thread_cond, thread_nms)

        if len(in_ids) == 0:
            return [], [], []
        return np.array(areas)[in_ids], np.array(conf)[in_ids], cls_id[in_ids]

    def infer_img(self, in_img0, thread_nms=0.4, thread_cond=0.5):
        # 图像预处理
        img = cv2.resize(in_img0, [self.model_w, self.model_h], interpolation=cv2.INTER_AREA)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        blob = np.expand_dims(np.transpose(img, (2, 0, 1)), axis=0)
        # 模型推理
        outs = self.net.run(None, {self.net.get_inputs()[0].name: blob})[0].squeeze(axis=0)
        # 输出坐标矫正
        outs = self.cal_outputs(outs)
        # 检测框计算
        img_h, img_w, _ = np.shape(in_img0)
        boxes, confs, out_ids = self.post_process_opencv(outs, img_h, img_w, thread_nms, thread_cond)
        return boxes, confs, out_ids


if __name__ == "__main__":
    Run_frame = OnnxFrame("./cross1.onnx")
    # 进行推理
    cap = VideoCapture(0)
    while 1:
        img0 = cap.read()[1]
        t1 = time.time()
        det_boxes, scores, ids = Run_frame.infer_img(img0, 0.4, 0.5)

        t2 = time.time()
        print("frame runtime: %.4f" % (t2 - t1))
        # 结果绘图
        for box, score, id1 in zip(det_boxes, scores, ids):
            label = '%s:%.2f' % (Run_frame.dic_labels[id1], score)

            Run_frame.plot_one_box(box.astype(np.int), img0,
                    (0, 255, 0), in_label=label, line_thickness=None)

        cv2.imshow('img', img0)
        cv2.waitKey(1)


