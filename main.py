import glob
import cv2
import numpy as np
from ultralytics import YOLO
from python.算法.fish import *
from scipy.spatial import ConvexHull


# 定义鱼群类
class Shoal():
    def __init__(self):
        self.img = None
        self.fishes = [Fish() for i in range(5)]
        self.cenFish = Fish()
        self.fishList = []
        self.positions = []

    def loadImg(self, imgPath: str):
        img = cv2.imread(imgPath)
        # img = cv2.resize(img, (640, 480))
        self.img = img

    def draw(self):
        img = self.img
        for i in range(len(self.fishes)):
            # 绘制鱼的编号
            cv2.putText(img, str(i), (int(self.fishes[i].x), int(self.fishes[i].y -60)), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
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
            for _pos in _positions:
                cv2.circle(self.img,(_pos[0], _pos[1]), 2, (0, 0, 255), -1)
            # cv2.polylines(img, [_positions], False, (28, 55, 112), 1)
            # cv2.polylines(img, [np.array(self.hull_vertices)], True, (128, 255, 212), 2)
        _positions = np.array(self.cenFish.positions[-20:]).astype(np.int32)
        # _positions转化成整数
        # 鱼群中心轨迹
        cv2.polylines(img, [_positions], False, (28, 55, 112), 1)
        cv2.polylines(img, [np.array(self.hull_vertices)], True, (128, 255, 212), 2)

        # 鱼群重心
        cv2.circle(img, (int(self.center[0]), int(self.center[1])), 5, (96, 128, 255), -1)
        cv2.putText(img, f'IV: {self.cenFish.instantaneous_speed():.2f}',   # 瞬时鱼群速度
                    (int(self.center[0]), int(self.center[1] - 30)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (96, 128, 255),
                    2,
                    )
        cv2.putText(img, f'AV: {self.cenFish.average_speed():.2f}', # 平均鱼群速度
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
        yuGangMianJi = 500000 # 鱼缸面积，500000像素
        cv2.putText(img, f'FSD: {100*self.area/yuGangMianJi:.1f}%',  # 分散度
                    (int(self.center[0]), int(self.center[1] + 120)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (56, 88, 225),
                    2,
                    )
        return  img

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

        distances = [[]for i in range(len(fishList))]
        min_dist = float('inf')
        max_dist = 0
        # 计算fishList中，各点间的最大距离和最小距离和平均距离
        for i in range(len(fishList)):
            for j in range(len(fishList)):
                distances[i].append(np.linalg.norm(np.array(fishList[i]) - np.array(fishList[j])))
                if i!=j:
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

    def loadModel(self,modelPath):
        self.model = YOLO(modelPath)
    def detect(self):
        imgList = glob.glob(r"..\datasets\user\fishs\*.jpg")
        # 保存imgList里的内容，每行一个，到txt中
        model = self.model
        for imgPath in imgList:
            results = model.predict(source=imgPath, save=False, save_txt=False, show=False, conf=0.25, verbose=False)
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
            img = shoal.draw()
            # img = shoal.img
            cv2.imshow('img', img)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                break


if __name__ == '__main__':
    # Load a model
    # model = YOLO(
    #     r'..\model\best.pt')  # load a pretrained model (recommended for training)
    shoal = Shoal()  # 定义鱼群类
    shoal.loadModel(r'..\model\best.pt')
    shoal.detect()