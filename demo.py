from keras.models import load_model
from keras.optimizers import SGD
from models.flow_gated_models import *

from preprocess.Video2Numpy import *
import ultralytics
from ultralytics import YOLO
import os
import sys
import time
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from ultralytics import YOLO
from PyQt6 import QtCore, QtGui, QtWidgets, uic
from PyQt6.QtWidgets import *
from PyQt6.uic import loadUi
from PyQt6.QtCore import QThread, pyqtSignal
from PyQt6.QtGui import QPixmap
import PyQt6.QtCore as Qt

class Main(QMainWindow):
    def __init__(self):
        super(Main, self).__init__()
        uic.loadUi('ui/demo.ui', self)
        
        self.real_path = ""  # Thêm khởi tạo cho real_path
        self.pred_path = ""  # Thêm khởi tạo cho pred_path
        
        
        self.open_vid.clicked.connect(self.output_vid.clear)
        self.open_vid.clicked.connect(self.input_vid.clear)
        self.open_vid.clicked.connect(self.openvid)
        self.fgm_bt.clicked.connect(self.fgm_pred)
        self.yolo_bt.clicked.connect(self.yolo_pred)
        self.play_vid.clicked.connect(self.start_play_video)
        self.thread = {}
        
    def start_play_video(self):
        self.thread[1] = capture_video(index=1, path=self.real_path[0])
        self.thread[2] = capture_video(index=2, path=self.pred_path)
        self.thread[1].start()
        self.thread[2].start()
        self.thread[1].signal.connect(self.show_input_video)
        self.thread[2].signal.connect(self.show_output_video)

        
    def fgm_pred(self):
        fgm = FlowGatedModels().build_model()
        fgm.load_weights('./weights/best_models_fgm.h5')
        FGM_video_results(fgm, self.real_path[0])
        self.pred_path = './outputs/outputs_video/FGM_outputs.avi'
        self.show_first_frame(self.pred_path, self.output_vid)

    def yolo_pred(self):
        yolo = YOLO('./weights/best_yolov8.pt')
        
        YOLOv8_video_results(yolo, self.real_path[0])
        self.pred_path = './outputs/outputs_video/YOLOv8_outputs.avi'
        self.show_first_frame(self.pred_path, self.output_vid)
        
    def openvid(self):
        self.real_path = QFileDialog.getOpenFileName(None, 'Mở File',' ','(*.avi);;(*.mp4)')
        if self.real_path[0]:  # Nếu người dùng chọn một file
            self.show_first_frame(self.real_path[0], self.input_vid)

    def show_first_frame(self, path, vid):
        # Mở video và đọc khung hình đầu tiên
        cap = cv2.VideoCapture(path)
        ret, first_frame = cap.read()
        if ret:
            qt_img = self.convert_cv_qt(first_frame)
            vid.setPixmap(qt_img)
        cap.release()
        
    def show_input_video(self, cv_img):
        """Updates the image_label with a new opencv image"""
        qt_img = self.convert_cv_qt(cv_img)
        self.input_vid.setPixmap(qt_img)
        
    def show_output_video(self, cv_img):
        """Updates the image_label with a new opencv image"""
        qt_img = self.convert_cv_qt(cv_img)
        self.output_vid.setPixmap(qt_img)
        
    def stop_capture_video(self):
        self.thread[1].stop()
        self.thread[2].stop()
        
        
    def convert_cv_qt(self, cv_img):
        rgb_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_img.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QtGui.QImage(rgb_img.data, w, h, bytes_per_line, QtGui.QImage.Format.Format_RGB888)
        p = convert_to_Qt_format.scaled(320, 240, Qt.Qt.AspectRatioMode.KeepAspectRatio)
        return QPixmap.fromImage(p)


class capture_video(QThread):
    signal = pyqtSignal(np.ndarray)
    def __init__(self, index, path):
        self.index = index
        self.path = path
        print("start threading", self.index)
        super(capture_video, self).__init__()

    def run(self):
        cap = cv2.VideoCapture(self.path)
        fps = cap.get(cv2.CAP_PROP_FPS)  # Lấy tốc độ khung hình của video
        while True:
            ret, cv_img = cap.read()
            if ret:
                self.signal.emit(cv_img)
                time.sleep(1/fps)  # Dừng lại một khoảng thời gian để đạt được tốc độ khung hình mong muốn
            else:
                break  # Nếu không còn khung hình nào, kết thúc luồng

    def stop(self):
        print("stop threading", self.index)
        self.terminate()
        
        
if __name__=='__main__':
    app = QApplication(sys.argv)
    widgets = QtWidgets.QStackedWidget()
    
    mainw = Main()
    
    mainw.show()
    app.exec()