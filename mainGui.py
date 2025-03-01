# coding:utf-8
import os
import sys

from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QIcon, QColor, QPixmap, QImage
from PyQt5.QtWidgets import QApplication, QFileDialog, QTableWidgetItem
from qfluentwidgets import (NavigationItemPosition, SplitFluentWindow, FluentTranslator, isDarkTheme)
from qfluentwidgets import FluentIcon as FIF

from python.GUI.widget.widegt_load import Ui_load_widget
from python.GUI.widget.widegt_save import Ui_save_widget
from python.GUI.widget.widegt_show import Ui_show_widget
from python.GUI.widget.widegt_setting import Ui_setting_widget
from python.GUI.widget.empty_widget import Widget
import glob
import cv2
import numpy as np
from ultralytics import YOLO
from python.算法.fish import *
from scipy.spatial import ConvexHull
import pandas as pd


class Shoal():
    def __init__(self):
        self.img = None
        self.fishes = [Fish() for i in range(5)]
        self.cenFish = Fish()
        self.fishList = []
        self.positions = []
        self.单鱼历史位置 = [[] for i in range(5)]
        self.单鱼历史瞬时速度 = [[] for i in range(5)]
        self.单鱼历史平均速度 = [[] for i in range(5)]
        self.单鱼历史瞬时角速度 = [[] for i in range(5)]

        self.鱼群历史重心位置 = []
        self.鱼群历史重心瞬时速度 = []
        self.鱼群历史重心平均速度 = []
        self.鱼群历史分散度 = []
        self.鱼群历史瞬时最近距离 = []
        self.鱼群历史瞬时最远距离 = []

    def loadImg(self, imgPath: str):
        img = cv2.imread(imgPath)
        self.img = img

    def draw(self):
        img = self.img
        for i in range(len(self.fishes)):
            # 绘制鱼的编号
            cv2.putText(img, str(i), (int(self.fishes[i].x), int(self.fishes[i].y - 60)), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 0, 255), 2)
            # 绘制鱼的平均速度
            cv2.putText(img, f"AV:{self.fishes[i].average_speed():.1f}",
                        (int(self.fishes[i].x), int(self.fishes[i].y - 40)), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 0, 255), 2)
            # 绘制鱼的瞬时速度
            cv2.putText(img, f"IV:{self.fishes[i].instantaneous_speed():.1f}",
                        (int(self.fishes[i].x), int(self.fishes[i].y - 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 0, 255), 2)
            # 绘制鱼的角速度
            # 计算极坐标的角度（以弧度表示）
            angle_radians = np.arctan2(self.fishes[i].y - self.center[1], self.fishes[i].x - self.center[0])
            # angle_degrees = np.degrees(angle_radians)
            cv2.putText(img, f"AAV:{angle_radians:.1f}",
                        (int(self.fishes[i].x), int(self.fishes[i].y + 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 0, 255), 2)
            # 单条鱼位置
            cv2.circle(self.img, (int(self.fishes[i].x), int(self.fishes[i].y)), 5, (0, 0, 255), -1)
            _positions = np.array(self.fishes[i].positions[-20:]).astype(np.int32)
            # 找到第一个不为零的行的索引
            nonzero_indices = np.any(_positions != 0, axis=1)
            first_nonzero_index = np.argmax(nonzero_indices)

            # 保留第一个不为零的行及其之后的所有行
            _positions = _positions[first_nonzero_index:]

            for _pos in _positions:
                cv2.circle(self.img, (_pos[0], _pos[1]), 2, (0, 0, 255), -1)
            # cv2.polylines(img, [_positions], False, (28, 55, 112), 1)
            # cv2.polylines(img, [np.array(self.hull_vertices)], True, (128, 255, 212), 2)
        _positions = np.array(self.cenFish.positions[-20:]).astype(np.int32)
        # 找到第一个不为零的行的索引
        nonzero_indices = np.any(_positions != 0, axis=1)
        first_nonzero_index = np.argmax(nonzero_indices)

        # 保留第一个不为零的行及其之后的所有行
        _positions = _positions[first_nonzero_index:]

        # _positions转化成整数
        # 鱼群中心轨迹
        cv2.polylines(img, [_positions], False, (28, 55, 112), 1)
        cv2.polylines(img, [np.array(self.hull_vertices)], True, (128, 255, 212), 2)

        # 鱼群重心
        cv2.circle(img, (int(self.center[0]), int(self.center[1])), 5, (96, 128, 255), -1)
        cv2.putText(img, f'IV: {self.cenFish.instantaneous_speed():.2f}',  # 瞬时鱼群速度
                    (int(self.center[0]), int(self.center[1] - 30)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (96, 128, 255),
                    2,
                    )
        cv2.putText(img, f'AV: {self.cenFish.average_speed():.2f}',  # 平均鱼群速度
                    (int(self.center[0]), int(self.center[1] - 60)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (96, 128, 255),
                    2,
                    )
        cv2.putText(img, f'avg_dist: {self.avg_dist:.2f}',  # 某一帧平均距离
                    (int(self.center[0]), int(self.center[1] + 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (96, 128, 255),
                    2,
                    )
        cv2.putText(img, f'max_dist: {self.max_dist:.2f}',  # 某一帧瞬时最远距离
                    (int(self.center[0]), int(self.center[1] + 60)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (76, 108, 235),
                    2,
                    )
        cv2.putText(img, f'min_dist: {self.min_dist:.2f}',  # 某一帧瞬时最近邻距离
                    (int(self.center[0]), int(self.center[1] + 90)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (76, 108, 235),
                    2,
                    )
        #
        yuGangMianJi = 500000  # 鱼缸面积，500000像素
        cv2.putText(img, f'FSD: {100 * self.area / yuGangMianJi:.1f}%',  # 分散度
                    (int(self.center[0]), int(self.center[1] + 120)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (56, 88, 225),
                    2,
                    )
        # =============================================================================================
        # 保存数据
        for i in range(len(self.fishes)):
            self.单鱼历史位置[i].append([self.fishes[i].x, self.fishes[i].y])
            self.单鱼历史瞬时速度[i].append(self.fishes[i].instantaneous_speed())
            self.单鱼历史平均速度[i].append(self.fishes[i].average_speed())
            angle_radians = np.arctan2(self.fishes[i].y - self.center[1], self.fishes[i].x - self.center[0])
            self.单鱼历史瞬时角速度[i].append(angle_radians)

        self.鱼群历史重心位置.append([self.center[0], self.center[1]])
        self.鱼群历史重心瞬时速度.append(self.cenFish.instantaneous_speed())
        self.鱼群历史重心平均速度.append(self.cenFish.average_speed())
        self.鱼群历史分散度.append(100 * self.area / yuGangMianJi)
        self.鱼群历史瞬时最近距离.append(self.min_dist)
        self.鱼群历史瞬时最远距离.append(self.max_dist)

        return img

    def move(self, fishList, t):
        fishBoxList = np.array(fishList)
        # 找到凸多边形的凸包（Convex Hull）
        hull = ConvexHull(fishBoxList)
        hull_vertices = fishBoxList[hull.vertices]
        # 计算凸多边形hull_vertices的面积
        self.area = cv2.contourArea(hull_vertices)
        # 找到凸多边形的中心点
        self.center = np.mean(fishBoxList, axis=0)
        self.cenFish.move(self.center[0], self.center[1], t)
        self.positions.append(self.center)
        # 取最后30个位置
        if len(self.positions) > 30:
            self.positions = self.positions[-30:]

        self.hull_vertices = hull_vertices

        # 对鱼群进行排序，尽量和上一次的鱼群能对应
        fishList = self.sortFishList(fishList)

        distances = [[] for i in range(len(fishList))]
        min_dist = float('inf')
        max_dist = 0
        # 计算fishList中，各点间的最大距离和最小距离和平均距离
        for i in range(len(fishList)):
            for j in range(len(fishList)):
                distances[i].append(np.linalg.norm(np.array(fishList[i]) - np.array(fishList[j])))
                if i != j:
                    if distances[i][-1] < min_dist:
                        min_dist = distances[i][-1]
                    if distances[i][-1] > max_dist:
                        max_dist = distances[i][-1]
        # print(sum(sum(t) for t in distances) / (len(distances) ** 2 - len(distances)))
        # print(f"min_dist:{min_dist:.2f}, max_dist:{max_dist:.2f}")
        self.avg_dist = sum(sum(t) for t in distances) / (len(distances) ** 2 - len(distances))
        self.min_dist = min_dist
        self.max_dist = max_dist

        for i in range(len(fishList)):
            self.fishes[i].move(fishList[i][0], fishList[i][1], t)

        # self.draw()

    # 用位置把每条鱼和上一帧的同一条进行匹配
    def sortFishList(self, fishList):
        if self.fishList == []:
            self.fishList = fishList
            return fishList
        # 找到每个 b 中的坐标与 a 中坐标的对应关系
        correspondences = []
        used_a_points = set()  # 记录已经匹配的 a 中的点
        for point_b in self.fishList:
            min_dist = float('inf')

            closest_point = None
            for point_a in fishList:
                point_a = tuple(point_a)
                if point_a in used_a_points:
                    continue  # 如果点已经匹配过，则跳过
                # 计算欧氏距离
                dist = np.linalg.norm(np.array(point_b) - np.array(point_a))
                if dist < min_dist:
                    min_dist = dist
                    closest_point = point_a
            if closest_point:
                correspondences.append(closest_point)
                used_a_points.add(closest_point)

        return correspondences

    def loadModel(self, modelPath):
        self.model = YOLO(modelPath)

    def detect(self):
        imgList = glob.glob(r"..\datasets\test\*.jpg")
        t = 1
        model = self.model
        for imgPath in imgList:
            results = model.predict(source=imgPath, show=False, conf=0.25, verbose=False)
            result = results[0]

            boxes = result.boxes.xywh.int().cpu().tolist()
            cls = result.boxes.cls.int().cpu().tolist()
            name = model.names

            # img = cv2.imread(imgPath)
            shoal.loadImg(imgPath)
            fishBoxList = []

            for i in range(len(boxes)):
                if name[cls[i]] == 'fish':
                    fishBoxList.append([boxes[i][0], boxes[i][1]])

            if len(fishBoxList) == 5:
                t = 1
                _fishBoxList = fishBoxList
            else:
                t += 1
                fishBoxList = _fishBoxList

            shoal.move(fishBoxList, t)
            self.outimg = shoal.draw()
            # img = shoal.img
            # cv2.imshow('img', img)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                break


class Window(SplitFluentWindow, Shoal):
    def __init__(self):
        self.isMicaEnabled = False
        super().__init__()
        self.Ui_load_widget = Ui_load_widget(self)
        # self.最小曲率法计算插值 = Ui_calAnyM_widget(self)
        # self.Ui_setting_widget = Ui_save_widget(self)
        self.Ui_show_widget = Ui_show_widget(self)
        self.Ui_save_widget = Ui_save_widget(self)
        self.Ui_load_widget.PrimaryPushButton.clicked.connect(self.startDetect)

        # self.Ui_load_widget.clicked.connect(self.startTimer)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.timerEvent)

        self.initNavigation()
        self.initWindow()

        self.fishBoxList = []
        self._fishBoxList = []
        self.fisheExcel = {}
        self.Ui_save_widget.PrimaryPushButton.clicked.connect(self.outputExcel)
        self.settingOpenDirs()

    def outputExcel(self):

        # 创建一个数据框
        df = pd.DataFrame(self.fisheExcel)
        excelPath = os.path.join(self.parent_directory, "output/excel/data.xlsx")
        # 写入到 Excel 文件
        df.to_excel(excelPath, index=False)
        img = shoal.img
        # 绘制每一条鱼的轨迹
        # zeroBackground = np.zeros_like()
        img_height, img_width, _ = img.shape
        # 创建一个全黑图像
        black_img = np.zeros((img_height, img_width, 3), dtype=np.uint8)  # 如果是灰度图像，只需使用 (img_height, img_width) 即可
        # 创建一个全白图像
        white_img = np.zeros((img_height, img_width, 3), dtype=np.uint8)
        white_img = white_img + 255
        # 定义一些颜色
        colors = [
            (255, 0, 0),  # 蓝色
            (0, 255, 0),  # 绿色
            (0, 0, 255),  # 红色
            (255, 255, 0),  # 青色
            (255, 0, 255),  # 洋红
            (0, 255, 255)  # 黄色
        ]
        for i in range(5):
            _black_img = black_img.copy()
            # 在black_img上绘制顶点
            _fishPos = self.fisheExcel[f"鱼{i}位置"]
            _fishPos = np.array(_fishPos, dtype=np.int32)
            # print(_fishPos)
            # 绘制多边形_fishPos在_black_img上
            cv2.polylines(_black_img, [_fishPos], False, (255, 255, 255), 2)
            cv2.polylines(white_img, [_fishPos], False, colors[i], 2)
            # cv2.imshow("fish"+str(i), _black_img)
            # cv2.waitKey(0)
            cv2.imwrite(os.path.join(self.parent_directory, "output/imgs/" + str(i) + ".jpg"), _black_img)
        cv2.imwrite(os.path.join(self.parent_directory, "output/imgs/fishes.jpg"), white_img)

        # 设置行列数
        self.Ui_save_widget.TableWidget.setRowCount(df.shape[0])  # 行数
        self.Ui_save_widget.TableWidget.setColumnCount(df.shape[1])  # 列数

        # 设置表头
        self.Ui_save_widget.TableWidget.setHorizontalHeaderLabels(df.columns.tolist())

        # 加载数据到 TableWidget
        for i in range(df.shape[0]):
            for j in range(df.shape[1]):
                self.Ui_save_widget.TableWidget.setItem(i, j, QTableWidgetItem(str(df.iloc[i, j])))

    def settingOpenDirs(self):
        dirDir = {
            "根目录": r"",
            "模型文件夹": r"model",
            "输出文件夹": r"output",
            "文档文件夹": r"docx",
            "数据文件夹": r"datasets",
        }

        # 获取当前文件所在的目录（文件路径的目录部分）
        self.current_directory = os.getcwd()
        self.parent_directory = os.path.dirname(self.current_directory)

        self.Ui_load_widget.ElevatedCardWidget.clicked.connect(lambda: self.open_in_file_explorer(dirDir["根目录"]))
        self.Ui_load_widget.ElevatedCardWidget_2.clicked.connect(
            lambda: self.open_in_file_explorer(dirDir["输出文件夹"]))
        self.Ui_load_widget.ElevatedCardWidget_3.clicked.connect(
            lambda: self.open_in_file_explorer(dirDir["模型文件夹"]))
        self.Ui_load_widget.ElevatedCardWidget_4.clicked.connect(
            lambda: self.open_in_file_explorer(dirDir["文档文件夹"]))
        self.Ui_load_widget.ElevatedCardWidget_5.clicked.connect(
            lambda: self.open_in_file_explorer(dirDir["数据文件夹"]))

    def initNavigation(self):
        # add sub interface
        self.addSubInterface(self.Ui_load_widget, FIF.HOME, '开始')
        # self.addSubInterface(self.Ui_setting_widget, FIF.ALBUM, 'setting', NavigationItemPosition.SCROLL)
        self.addSubInterface(self.Ui_save_widget, FIF.ADD, 'save', NavigationItemPosition.SCROLL)
        self.addSubInterface(self.Ui_show_widget, FIF.PENCIL_INK, 'show', NavigationItemPosition.SCROLL)

        self.navigationInterface.addItem(
            routeKey='settingInterface',
            icon=FIF.SETTING,
            text='设置',
            position=NavigationItemPosition.BOTTOM,
        )
        self.navigationInterface.setCurrentItem("数据输入")

    def initWindow(self):
        self.resize(900, 700)
        self.setWindowIcon(QIcon(':/qfluentwidgets/images/logo.png'))
        self.setWindowTitle('  ')
        desktop = QApplication.desktop().availableGeometry()
        w, h = desktop.width(), desktop.height()
        self.move(w // 2 - self.width() // 2, h // 2 - self.height() // 2)

    def startDetect(self):
        self.loadModel(r'..\model\best.pt')
        # self.imgList = glob.glob(r"..\datasets\user\fishs\*.jpg")
        folder_path = self.open_folder_dialog()
        if folder_path:

            self.imgList = glob.glob(os.path.join(folder_path, '*.jpg'))
            print(os.path.join(folder_path, '*.jpg'))
            # self.imgList = glob.glob(r"..\datasets\test\*.jpg")
            self.timer.start(25)  # 在这里设置检测延迟
            self.stratDetectNum = 0
        else:
            print("未正确选择文件夹")
            return

    def open_folder_dialog(self):
        folder_path = QFileDialog.getExistingDirectory(self, '选择鱼群图片存放的文件夹')
        if folder_path:
            print(f"选择的文件夹是：{folder_path}")
            return folder_path
        else:
            return False

    def timerEvent(self):
        if self.stratDetectNum >= len(self.imgList):
            self.outputExcel()
            self.timer.timeout()
            return
        # model = self.model
        imgPath = self.imgList[self.stratDetectNum]
        self.stratDetectNum += 1
        try:
            cv2.imread(imgPath)
        except:
            self.outputExcel()
            self.timer.timeout()
            return
        results = self.model.predict(source=imgPath, save=False, save_txt=False, show=False, conf=0.25, verbose=False)
        result = results[0]

        boxes = result.boxes.xywh.int().cpu().tolist()
        cls = result.boxes.cls.int().cpu().tolist()
        name = self.model.names
        shoal.loadImg(imgPath)
        fishBoxList = []
        for i in range(len(boxes)):
            if name[cls[i]] == 'fish':
                fishBoxList.append([boxes[i][0], boxes[i][1]])
        global t
        if len(fishBoxList) == 5:
            t = 1
        elif len(fishBoxList) > 5:  # 如果这一帧中检测超过了5条鱼，只取前五条
            t = 1
            fishBoxList = fishBoxList[0:5]
        else:  # 如果不足五条鱼，则认为检测不准确，跳过这一帧，在下一帧设定用时加一。
            if t is None:  # 如果全局变量尚未赋值
                t = 1  # 赋值为1
            else:
                t += 1
            return
        shoal.move(fishBoxList, t)  # 输入鱼群坐标信息，和时间信息
        self.outimg = shoal.draw()
        height, width, channel = self.outimg.shape
        bytes_per_line = 3 * width
        q_img = QImage(self.outimg.data, width, height, bytes_per_line, QImage.Format_RGB888)
        q_img = q_img.rgbSwapped()  # Qt与OpenCV的颜色通道顺序不同，需要交换通道

        # 将QImage转换为QPixmap
        pixmap = QPixmap.fromImage(q_img)
        self.Ui_show_widget.label.setPixmap(pixmap)
        # ====================================================================================
        # 保存数据
        for i in range(len(fishBoxList)):
            self.fisheExcel[f"鱼{i}位置"] = shoal.单鱼历史位置[i]
            self.fisheExcel[f"鱼{i}瞬时速度"] = shoal.单鱼历史瞬时速度[i]
            self.fisheExcel[f"鱼{i}平均速度"] = shoal.单鱼历史平均速度[i]
            self.fisheExcel[f"鱼{i}角速度"] = shoal.单鱼历史瞬时角速度[i]
        self.fisheExcel["鱼群重心位置"] = shoal.鱼群历史重心位置
        self.fisheExcel["鱼群瞬时位置"] = shoal.鱼群历史重心瞬时速度
        self.fisheExcel["鱼群平均速度"] = shoal.鱼群历史重心平均速度
        self.fisheExcel["鱼群分散度"] = shoal.鱼群历史分散度
        self.fisheExcel["鱼群最远距离"] = shoal.鱼群历史瞬时最远距离
        self.fisheExcel["鱼群最近距离"] = shoal.鱼群历史瞬时最近距离
        # print(self.fisheExcel)

    def open_in_file_explorer(self, folder_path):
        folder_path = os.path.join(self.parent_directory, folder_path)
        try:
            os.startfile(folder_path)
        except Exception as e:
            print("无法打开文件夹:", e)


if __name__ == '__main__':
    # 鱼群
    shoal = Shoal()
    # 界面
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps)
    app = QApplication(sys.argv)
    w = Window()
    # ===============================================================
    # 程序入口
    w.show()
    # ===============================================================
    # 结束，内存回收
    app.exec_()
