import os
import numpy as np
import torch
from PyQt5 import *
import sys
import cv2
from PyQt5.QtWidgets import (QApplication, QMainWindow, QFileDialog, QMessageBox)
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QTimer, Qt
from PIL import Image
from my_ui_pro import Ui_MainWindow

#
# 2022212026-刘永林
# 占用
#

YOLO_PATH = r'Z:\yolo_git\yolov5-master'
model = torch.hub.load(YOLO_PATH, 'custom', path=YOLO_PATH + r'\yolov5s.pt', source='local')
class CameraApp(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.detect_status = False
        self.quwu_status = False
        self.cap = None
        self.timer = QTimer()
        self.timer01 = QTimer()
        self.video_length = 0  # 视频时长（帧数）
        self.detecting = False  # 检测标志位
        self.quwuing = False
        # 连接按钮事件
        self.start_Button.clicked.connect(self.start_camera)
        self.stop_Button.clicked.connect(self.stop_camera)
        self.stop_Button.setEnabled(False)
        self.load_button.clicked.connect(self.load_video)
        self.capture_button.clicked.connect(self.capture_image)
        self.capture_button.setEnabled(False)
        self.record_button.clicked.connect(self.start_recording)
        self.record_button.setEnabled(False)
        self.stop_record_button.clicked.connect(self.stop_recording)
        self.stop_record_button.setEnabled(False)
        self.exit_button.clicked.connect(self.close)
        self.horizontalSlider.valueChanged.connect(self.set_video_position)
        self.horizontalSlider.setVisible(False)
        self.timer.timeout.connect(self.update_frame)

        self.detect_button.clicked.connect(self.detect_object)
        self.detect_button.setEnabled(False)
        self.detect_button_2.clicked.connect(self.quwu_object)
        self.detect_button_2.setEnabled(False)

    def show_warning(self, message):
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Warning)
        msg.setText("警告：" + message)
        msg.setWindowTitle("警告")
        msg.setStandardButtons(QMessageBox.Ok)
        msg.exec_()

    def start_camera(self):
        self.detect_button.setEnabled(True)
        self.detect_button_2.setEnabled(True)
        self.horizontalSlider.setVisible(False)
        # VideoCapture对象，cv中两种来源
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            self.show_warning("无法打开摄像头。")
            return
        self.timer.start(20)# ms
        self.record_button.setEnabled(True)
        self.stop_Button.setEnabled(True)
        self.capture_button.setEnabled(True)

    def stop_camera(self):
        self.horizontalSlider.setVisible(False)
        self.timer.stop()
        if self.cap is not None:
            self.cap.release()
            self.cap = None
            self.capture_button.setEnabled(False)
            self.record_button.setEnabled(False)
            self.stop_Button.setEnabled(False)
        self.label.clear()

    def set_video_position(self, position):
        if self.cap is not None:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, position)  # 设置视频播放位置
    def load_video(self):
        video_file, _ = QFileDialog.getOpenFileName(self, "选择视频文件", "", "Video Files (*.mp4 *.avi *.mov *.mkv)")
        if video_file:
            self.cap = cv2.VideoCapture(video_file)
            if not self.cap.isOpened():
                self.show_warning("无法打开视频文件")
                return
            self.horizontalSlider.setVisible(True)
            self.video_length = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.horizontalSlider.setRange(0, self.video_length)
            self.timer.start(20) # 防止第一下就加载而timer没有开启
            self.record_button.setEnabled(False)
            self.capture_button.setEnabled(False)
            self.stop_Button.setEnabled(False)

    def update_frame(self):
        if self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                if self.detect_status:
                    frame=self.run_yolo_detection(frame)
                if self.quwu_status:
                    frame = self.quwu_function(frame)
                else:
                    rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    h, w, ch = rgb_image.shape
                    bytes_per_line = ch * w
                    q_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
                    self.label.setPixmap(QPixmap.fromImage(q_image))
                    # Pixmap常用于显示，QPixmap常用于存储和操作
                if self.horizontalSlider.isVisible():
                    current_frame = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
                    self.horizontalSlider.setValue(current_frame)

    def capture_image(self):
        if self.cap is not None:
            ret, frame = self.cap.read()
            if ret:
                if self.detect_status:  # 如果开启了识别
                    # fame是一个numpy的数组
                    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    results = model(img)  # YOLO 检测
                    results.render()  # 在图像上绘制识别框
                    frame = np.array(results.ims[0])  # 获取带框的图像
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # 转回 BGR 格式
                if self.quwu_status:
                    frame=self.quwu_function(frame)
                cv2.imwrite('captured_image.jpg', frame)  # 保存带有识别框的图像
                self.show_warning("图片已保存为captured_image.jpg")
        else:
            self.show_warning("未开启摄像头")

    def start_recording(self):
        if self.cap is not None:
            self.fourcc = cv2.VideoWriter_fourcc(*'XVID')# AVID也可以
            self.out = cv2.VideoWriter('recording.avi', self.fourcc, 50.0, (640, 480))
            self.timer01.timeout.connect(self.record_frame)
            self.timer01.start(20)
            self.record_status = True
            self.record_button.setEnabled(False)
            self.show_warning("开始录制")
            self.stop_record_button.setEnabled(True)
        else:
            self.show_warning("未开启摄像头")

    def record_frame(self):
        if self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                if self.detect_status:  # 如果开启了识别
                    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    results = model(img)  # YOLO 检测
                    results.render()  # 在图像上绘制识别框
                    frame = np.array(results.ims[0])  # 获取带框的图像
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # 转回 BGR 格式
                if self.quwu_status:
                    frame = self.quwu_function(frame)
                self.out.write(frame)  # 保存带有识别框的帧

    def stop_recording(self):
        if hasattr(self, 'out'):
            self.out.release()
            self.timer01.timeout.disconnect(self.record_frame)
            self.timer01.stop()
            self.record_button.setEnabled(False)
            self.record_status = False
            self.record_button.setEnabled(True)
            self.show_warning("录制完成！文件保存为：recording.avi")
            self.stop_record_button.setEnabled(False)

    def detect_object(self):
        self.detect_status = not self.detect_status
        if self.detect_status:
            self.detect_button.setText("关闭识别")
        else:
            self.detect_button.setText("识别")

    def run_yolo_detection(self, frame):
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        results = model(img)
        results.render()
        result_image = np.array(results.ims[0])
        result_image = cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR)
        result_pixmap = self.convert_cv_qt(result_image)
        self.label.setPixmap(result_pixmap)
        return result_image

    def convert_cv_qt(self, cv_img):
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        return QPixmap.fromImage(convert_to_Qt_format).scaled(640, 480, Qt.KeepAspectRatio)
        # 保持宽高比
    def quwu_function(self, frame):
        print("quwu_function")
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.addWeighted(frame, 1.5, np.zeros(frame.shape, frame.dtype), 0, -50)
        # 1.5倍亮度
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        result_pixmap = self.convert_cv_qt(frame)
        self.label.setPixmap(result_pixmap)
        return frame


    def quwu_object(self):
        print("quwu_object")
        self.quwu_status = not self.quwu_status
        if self.quwu_status:
            self.detect_button_2.setText("关闭去雾")
        else:
            self.detect_button_2.setText("去雾")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = CameraApp()
    window.show()
    sys.exit(app.exec_())
