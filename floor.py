# python-38
# ------------------------------------------------
# 作者     :刘想
# 时间     :2023/1/4 15:20
# ------------------------------------------------
import logging

import cv2 as cv
import numpy as np
from skimage import morphology
import math
import json


class Peak_type:
    left_up = 0
    right_up = 1
    left_down = 2
    right_down = 3
    in_left_up = 4
    in_right_up = 5
    in_left_down = 6
    in_right_down = 7


# 枚举每一个种类
class img_type:
    exterior_wall = 0  # 外墙
    interior_wall = 1  # 内墙
    front_door = 2  # 前门
    interior_door = 3  # 内部门
    living_room = 4  # 客厅
    master_room = 5  # 主卧
    kitchen = 6  # 厨房
    bath_room = 7  # 浴室
    dining_room = 8  # 餐厅
    child_room = 9  # 儿童房
    study_room = 10  # 书房
    second_room = 11  # 次卧
    guest_room = 12  # 客房
    balcony = 13  # 阳台
    entrance = 14  # 入户门
    storage = 15  # 储物间
    wall_in = 16  # 内墙
    external_area = 17  # 外部区域

    @staticmethod
    def get_str(p_type: int):
        if p_type == img_type.exterior_wall:
            return "exterior_wall"
        elif p_type == img_type.interior_wall:
            return "interior_wall"
        elif p_type == img_type.front_door:
            return "front_door"
        elif p_type == img_type.interior_door:
            return "interior_door"
        elif p_type == img_type.living_room:
            return "living_room"
        elif p_type == img_type.master_room:
            return "master_room"
        elif p_type == img_type.kitchen:
            return "kitchen"
        elif p_type == img_type.bath_room:
            return "bath_room"
        elif p_type == img_type.dining_room:
            return "dining_room"
        elif p_type == img_type.child_room:
            return "child_room"
        elif p_type == img_type.study_room:
            return "study_room"
        elif p_type == img_type.second_room:
            return "second_room"
        elif p_type == img_type.guest_room:
            return "guest_room"
        elif p_type == img_type.guest_room:
            return "guest_room"
        elif p_type == img_type.balcony:
            return "balcony"
        elif p_type == img_type.entrance:
            return "entrance"
        elif p_type == img_type.storage:
            return "storage"
        elif p_type == img_type.wall_in:
            return "wall_in"
        elif p_type == img_type.external_area:
            return "external_area"
        else:
            return "invalid type"

    # 获取该像素的类型
    @staticmethod
    def get_type(pixel):
        r = pixel[2]
        g = pixel[1]
        if r == 127 and g == 14:
            return img_type.exterior_wall
        elif r == 0 and g == 16:
            return img_type.interior_wall
        elif r == 255 and g == 15:
            return img_type.front_door
        elif r == 0 and g == 17:
            return img_type.interior_door
        elif g == 0:
            return img_type.living_room
        elif g == 1:
            return img_type.master_room
        elif g == 2:
            return img_type.kitchen
        elif g == 3:
            return img_type.bath_room
        elif g == 4:
            return img_type.dining_room
        elif g == 5:
            return img_type.child_room
        elif g == 6:
            return img_type.study_room
        elif g == 7:
            return img_type.second_room
        elif g == 8:
            return img_type.guest_room
        elif g == 9:
            return img_type.balcony
        elif g == 10:
            return img_type.entrance
        elif g == 11:
            return img_type.storage
        elif g == 12:
            return img_type.wall_in
        elif g == 13:
            return img_type.external_area
        else:
            return None


def get_peak(img, i, j):
    # 根据形状特征判断是否为顶点
    p1 = img[i - 1, j - 1]
    p2 = img[i, j - 1]
    p3 = img[i + 1, j - 1]
    p4 = img[i - 1, j]
    p5 = img[i, j]
    p6 = img[i + 1, j]
    p7 = img[i - 1, j + 1]
    p8 = img[i, j + 1]
    p9 = img[i + 1, j + 1]
    if p5 == p6 and p6 == p8 and p8 == p9 and p1 != p5 and p2 != p5 and p3 != p5 and p4 != p5 and p7 != p5 and p5 != 0:
        return True, Peak_type.left_up
    elif p4 == p5 and p5 == p7 and p7 == p8 and p1 != p5 and p2 != p5 and p3 != p5 and p6 != p5 and p9 != p5 and p5 != 0:
        return True, Peak_type.right_up
    elif p1 == p2 and p2 == p5 and p5 == p4 and p3 != p5 and p6 != p5 and p7 != p5 and p8 != p5 and p9 != p5 and p5 != 0:
        return True, Peak_type.right_down
    elif p2 == p3 and p3 == p6 and p6 == p5 and p1 != p5 and p4 != p5 and p7 != p5 and p8 != p5 and p9 != p5 and p5 != 0:
        return True, Peak_type.left_down
    elif p1 == p2 and p2 == p3 and p3 == p4 and p4 == p5 and p5 == p6 and p6 == p7 and p7 == p8 and p8 != p9 and p5 != 0:
        return True, Peak_type.in_right_down
    elif p1 == p2 and p2 == p3 and p3 == p4 and p4 == p5 and p5 == p6 and p6 == p9 and p9 == p8 and p7 != p8 and p5 != 0:
        return True, Peak_type.in_left_down
    elif p9 == p8 and p8 == p7 and p7 == p6 and p6 == p5 and p5 == p4 and p1 == p4 and p1 == p2 and p2 != p3 and p5 != 0:
        return True, Peak_type.in_right_up
    elif p9 == p8 and p8 == p7 and p7 == p6 and p6 == p5 and p5 == p4 and p4 == p3 and p3 == p2 and p2 != p1 and p5 != 0:
        return True, Peak_type.in_left_up
    else:
        return False, Peak_type.left_up


def get_nearest_in_array(x, array):
    output = array[0]
    min_abs = abs(array[0] - x)
    for i in range(1, len(array)):
        temp_abs = abs(array[i] - x)
        if temp_abs < min_abs:
            min_abs = temp_abs
            output = array[i]
    return output


class Point:
    def __init__(self, _x: int, _y: int):
        self.x = int(_x)
        self.y = int(_y)


class AABB:
    def __init__(self, _min_p: Point, _max_p: Point):
        self.min_p = _min_p
        self.max_p = _max_p


class Circle:
    def __init__(self, _radius: int, _center: Point):
        self.radius = int(_radius)
        self.center = _center


def get_AABB_collision(box1: AABB, box2: AABB):
    min_x = max(box2.min_p.x, box1.min_p.x)
    max_x = min(box2.max_p.x, box1.max_p.x)
    min_y = max(box2.min_p.y, box1.min_p.y)
    max_y = min(box2.max_p.y, box1.max_p.y)
    x_collision = min_x <= max_x
    y_collision = min_y <= max_y
    if x_collision and y_collision:
        return AABB(Point(min_x, min_y), Point(max_x, max_y))
    return None


class Bubble_img:
    def __init__(self, _width: int, _height: int, baseImage):
        # cv.connectedComponentsWithStats()
        self.__bubbles = dict()
        self.__bridges = list()
        self.__width = _width
        self.__height = _height
        self.__bubble_img = baseImage  # 白色背景
        self.__generated = False

    def addBubble(self, _circle: Circle, p_type: int, diff: int):
        if p_type not in self.__bubbles.keys():
            self.__bubbles[p_type] = dict()
        self.__bubbles[p_type][diff] = _circle

    def addBridge(self, p_type1: int, diff1: int, p_type2: int, diff2: int):
        if p_type1 in self.__bubbles.keys() and p_type2 in self.__bubbles.keys():
            self.__bridges.append([p_type1, diff1, p_type2, diff2])

    def GenerateImage(self):
        if self.__generated:
            return self.__bubble_img
        for bri in self.__bridges:
            [p_type1, diff1, p_type2, diff2] = bri
            p1 = self.__bubbles[p_type1][diff1].center
            p2 = self.__bubbles[p_type2][diff2].center
            cv.line(self.__bubble_img, (p1.y, p1.x), (p2.y, p2.x), (255, 193, 0), 5)
        for p_type in self.__bubbles.keys():
            for diff in self.__bubbles[p_type].keys():
                size = self.__bubbles[p_type][diff].radius
                center = self.__bubbles[p_type][diff].center
                cv.circle(self.__bubble_img, (center.y, center.x), int(size / 2), (68, 114, 197), -1)
        return self.__bubble_img


class Grid_img:
    def __init__(self, _wall_img, _thickness):
        self.__width, self.__height = _wall_img.shape
        self.__wall_img = _wall_img
        self.__thickness = _thickness

    def GenerateImage(self):
        output_img = 255 * np.ones((self.__width, self.__height, 3), np.uint8)
        for i in range(self.__height):
            for j in range(self.__width):
                if self.__wall_img[i][j] != 0:
                    output_img[i][j] = (0, 0, 0)
        for i in range(1, self.__width - 1):
            for j in range(1, self.__height - 1):
                is_peak, p_type = get_peak(self.__wall_img, i, j)
                if is_peak:
                    center = Point(i, j)
                    # cv.circle(output_img, (int(center.y), int(center.x)), int(1), 200, cv.FILLED)
                    epsilon = int(self.__thickness / 2)
                    if p_type == Peak_type.left_up or p_type == Peak_type.in_left_up:
                        center.x = i + epsilon
                        center.y = j + epsilon
                    elif p_type == Peak_type.right_up or p_type == Peak_type.in_right_up:
                        center.x = i - epsilon
                        center.y = j + epsilon
                    elif p_type == Peak_type.left_down or p_type == Peak_type.in_left_down:
                        center.x = i + epsilon
                        center.y = j - epsilon
                    elif p_type == Peak_type.right_down or p_type == Peak_type.in_right_down:
                        center.x = i - epsilon
                        center.y = j - epsilon
                    # output_img[int(center.x), int(center.y)] = 100
                    cv.circle(output_img, (int(center.y), int(center.x)), int(self.__thickness), (193, 255, 0), cv.FILLED)
        # cv.imwrite("test.bmp", output_img)
        return output_img


# 获取膨胀aabb
def get_expansion_AABB(min_x1, min_y1, max_x1, max_y1, min_x2, min_y2, max_x2, max_y2, thickness):
    min_x = max(min_x1, min_x2)
    max_x = min(max_x1, max_x2)
    min_y = max(min_y1, min_y2)
    max_y = min(max_y1, max_y2)
    if min_x > max_x:
        aabb1 = AABB(Point(min_x1 - thickness, min_y1), Point(max_x1 + thickness, max_y1))
        aabb2 = AABB(Point(min_x2 - thickness, min_y2), Point(max_x2 + thickness, max_y2))
        return aabb1, aabb2
    elif min_y > max_y:
        aabb1 = AABB(Point(min_x1, min_y1 - thickness), Point(max_x1, max_y1 + thickness))
        aabb2 = AABB(Point(min_x2, min_y2 - thickness), Point(max_x2, max_y2 + thickness))
        return aabb1, aabb2
    else:
        aabb1 = AABB(Point(min_x1, min_y1), Point(max_x1, max_y1))
        aabb2 = AABB(Point(min_x2, min_y2), Point(max_x2, max_y2))
        return aabb1, aabb2


class Floor_plan:
    def __init__(self, image: np.ndarray):
        # < img_type, <像素b通道, [size, binaries_index, sum_x, sum_y, min_x, min_y, max_x, max_y] > >
        self.statistic = dict()
        self.img = image.copy()
        self.width, self.height, self.channel = self.img.shape
        self.wall = None
        self.skeleton = None
        self.binaries = list()  # 该列表记录每一种颜色的二值图
        self.thickness = 0
        logging.info("initial floor plan")
        self.__run_statistic()

    def __run_statistic(self):
        # 统计每一种颜色的面积和所有坐标的和用于计算中心点
        # 再记录每一种颜色的二值图, 保存该颜色binary的索引
        for i in range(self.width):
            for j in range(self.height):
                pixel_type = img_type.get_type(self.img[i, j])
                diff = self.img[i][j][0]
                if pixel_type in self.statistic.keys() and diff in self.statistic[pixel_type].keys():
                    # 存在该像素相同的元素
                    [size, index, sum_x, sum_y, min_x, min_y, max_x, max_y] = self.statistic[pixel_type][diff]
                    min_x = min(min_x, j)
                    max_x = max(max_x, j)
                    min_y = min(min_y, i)
                    max_y = max(max_y, i)
                    self.binaries[index][i][j] = 255
                    self.statistic[pixel_type][diff] = [size + 1, index, sum_x + i, sum_y + j, min_x, min_y, max_x,
                                                        max_y]
                elif pixel_type in self.statistic.keys() and diff not in self.statistic[pixel_type].keys():
                    # 存在相同label但是另外一间房
                    self.binaries.append(np.zeros((self.width, self.height), np.uint8))
                    self.binaries[len(self.binaries) - 1][i][j] = 255
                    self.statistic[pixel_type][diff] = [1, len(self.binaries) - 1, i, j, j, i, j, i]
                else:
                    # 第一次就添加一个binary图像, 初始化统计数据
                    self.binaries.append(np.zeros((self.width, self.height), np.uint8))
                    self.binaries[len(self.binaries) - 1][i][j] = 255
                    self.statistic[pixel_type] = {diff: [1, len(self.binaries) - 1, i, j, j, i, j, i]}

    def __get_binary_img(self, binary_type: int):
        output = list()
        # 根据输入类型获取该类型的二值图
        if binary_type in self.statistic.keys():
            value_dict = self.statistic[binary_type]
            for key in value_dict.keys():
                index = value_dict[key][1]
                output.append(self.binaries[index])
        return output

    def get_binary_skeleton(self):
        # 获取墙体骨架
        self.wall[self.wall == 255] = 1
        skeleton = morphology.skeletonize(self.wall)
        self.wall[self.wall == 1] = 255
        skeleton = skeleton.astype(np.uint8) * 255
        # 矫正骨架, 统计横竖两个方向的直方图, 使用投影进行矫正
        # 横向的统计列表
        horizon = [0 for _ in range(0, self.height)]
        # 竖向的统计列表
        vertical = [0 for _ in range(0, self.width)]
        for i in range(0, self.width):
            for j in range(0, self.height):
                if skeleton[i, j] == 255:
                    horizon[j] = horizon[j] + 1
                    vertical[i] = vertical[i] + 1
        # plt.hist(vertical, bins=height)
        # plt.show()
        # 用阈值将峰值坐标取出来
        base_line_hor = list()
        base_line_ver = list()
        for i in range(0, self.height):
            if horizon[i] > 10:
                base_line_hor.append(i)
        for i in range(0, self.width):
            if vertical[i] > 10:
                base_line_ver.append(i)

        horizon_img = np.zeros(skeleton.shape, np.uint8)
        vertical_img = horizon_img.copy()
        # 遍历像素 每一个骨架像素进行横向竖向投影生成两张图
        for i in range(0, self.width):
            for j in range(0, self.height):
                if skeleton[i, j] == 255:
                    y = get_nearest_in_array(j, base_line_hor)
                    horizon_img[i, y] = 255
                    x = get_nearest_in_array(i, base_line_ver)
                    vertical_img[x, j] = 255
        output = cv.add(horizon_img, vertical_img)
        return output

    def __calc_wall_thickness(self):
        # 获取墙体厚度和膨胀后的图
        binary_size = 0
        skeleton_size = 0
        for i in range(0, self.width):
            for j in range(0, self.height):
                if self.wall[i, j] != 0:
                    binary_size = binary_size + 1
        for i in range(0, self.width):
            for j in range(0, self.height):
                if self.skeleton[i, j] != 0:
                    skeleton_size = skeleton_size + 1
        thickness = 1.0 * binary_size / skeleton_size
        # print(thickness)
        after_point = thickness - int(thickness)
        if after_point > 0.75:
            thickness = int(thickness) + 1
        else:
            thickness = int(thickness)
        expansion = self.skeleton.copy()
        for i in range(0, self.width):
            for j in range(0, self.height):
                if self.skeleton[i, j] == 255:
                    # 按照厚度进行膨胀
                    for x in range(i - int(thickness / 2), i + int(thickness / 2) + 1):
                        for y in range(j - int(thickness / 2), j + int(thickness / 2) + 1):
                            expansion[x, y] = 255
        return thickness, expansion

    def generate_bubble_img(self):
        # 获取边缘轮廓图, 用几何特征, p的四邻域内有黑色则置为1
        baseImg = self.__get_outline()
        bb_img = Bubble_img(self.width, self.height, baseImg)
        # 获取所有的门, connectedComponentsWithStats() https://zhuanlan.zhihu.com/p/101371934
        front_door = self.__get_binary_img(img_type.front_door)[0]
        interior_door = self.__get_binary_img(img_type.interior_door)[0]
        door_img = cv.add(front_door, interior_door)
        room_set = {4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15}
        # 添加气泡
        for p_type in self.statistic.keys():
            if p_type in room_set:
                for diff in self.statistic[p_type].keys():
                    [size, _, sum_x, sum_y, _, _, _, _] = self.statistic[p_type][diff]
                    center = Point(sum_x / size, sum_y / size)
                    bb_img.addBubble(Circle(int(math.pow(size / 3.1415926, 0.5)), center), p_type, diff)
        # 添加连接线
        # 获取每一个房间的包围盒, 两两包围盒扩大厚度距离, 判断是否相交, 若相交则判断相交区域是否覆盖门
        for p_type1 in self.statistic.keys():
            if p_type1 not in room_set:
                continue
            for diff1 in self.statistic[p_type1].keys():
                for p_type2 in self.statistic.keys():
                    if p_type2 not in room_set:
                        continue
                    for diff2 in self.statistic[p_type2].keys():
                        if p_type1 == p_type2 and diff1 == diff2:
                            continue
                        [_, _, _, _, min_x1, min_y1, max_x1, max_y1] = self.statistic[p_type1][diff1]
                        [_, _, _, _, min_x2, min_y2, max_x2, max_y2] = self.statistic[p_type2][diff2]
                        aabb1, aabb2 = get_expansion_AABB(min_x1, min_y1, max_x1, max_y1, min_x2, min_y2, max_x2,
                                                          max_y2, self.thickness)
                        collision = get_AABB_collision(aabb1, aabb2)
                        if collision is not None:
                            for i in range(collision.min_p.y, collision.max_p.y):
                                for j in range(collision.min_p.x, collision.max_p.x):
                                    p_type = img_type.get_type(self.img[i][j])
                                    if p_type == img_type.front_door or p_type == img_type.interior_door:
                                        bb_img.addBridge(p_type1, diff1, p_type2, diff2)
        # 生成气泡图
        bubble_image = bb_img.GenerateImage()
        return bubble_image

    def __serialize_result(self):
        retDict = dict()
        retDict["thickness"] = int(self.thickness)
        retDict["roomInfo"] = list()
        print(self.statistic)
        room_set = {4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15}
        for key in self.statistic.keys():
            if key not in room_set:
                continue
            newKey = img_type.get_str(key)
            for version in self.statistic[key]:
                infoList = self.statistic[key][version]
                tmpDict = dict()
                tmpDict["name"] = newKey+str(version)
                tmpDict["size"] = infoList[0]
                tmpDict["center"] = [infoList[2]/infoList[0], infoList[3]/infoList[0]]
                tmpDict["min"] = [infoList[4], infoList[5]]
                tmpDict["max"] = [infoList[6], infoList[7]]
                retDict["roomInfo"].append(tmpDict)

        return json.dumps(retDict)

    def __get_outline(self):
        exterior_area = self.__get_binary_img(img_type.external_area)[0]
        output = 255 * np.ones((self.width, self.height, 3), np.uint8)
        for i in range(self.width):
            for j in range(self.height):
                if exterior_area[i][j] == 0 and \
                        (exterior_area[i][j - 1] == 255 or exterior_area[i - 1][j] == 255
                         or exterior_area[i + 1][j] == 255 or exterior_area[i][j + 1] == 255):
                    for x in range(-1, 1):
                        for y in range(-1, 1):
                            output[i + x][j + y] = (0, 0, 0)
        return output

    def getResult(self):
        interior_wall = self.__get_binary_img(img_type.interior_wall)[0]
        exterior_wall = self.__get_binary_img(img_type.exterior_wall)[0]
        front_door = self.__get_binary_img(img_type.front_door)[0]
        interior_door = self.__get_binary_img(img_type.interior_door)[0]
        door = cv.add(front_door, interior_door)
        self.wall = cv.add(interior_wall, exterior_wall)
        self.wall = cv.add(self.wall, door)
        # 生成墙体骨架和获取墙体厚度
        self.skeleton = self.get_binary_skeleton()
        self.thickness, expansion = self.__calc_wall_thickness()
        # 输入墙体图生成网格图
        grid = Grid_img(self.wall, self.thickness)
        grid_img = grid.GenerateImage()
        # 生成气泡图
        bubble_img = self.generate_bubble_img()
        # cv.waitKey(0)
        # cv.destroyAllWindows()
        retStr = self.__serialize_result()
        return retStr, grid_img, bubble_img

    def run(self):
        # 原图
        cv.namedWindow("Image")

        cv.imshow("Image", self.img)
        # _, interior_wall = self.get_binary_with_type(img_type.interior_wall)
        interior_wall = self.__get_binary_img(img_type.interior_wall)[0]
        exterior_wall = self.__get_binary_img(img_type.exterior_wall)[0]
        front_door = self.__get_binary_img(img_type.front_door)[0]
        interior_door = self.__get_binary_img(img_type.interior_door)[0]

        door = cv.add(front_door, interior_door)
        self.wall = cv.add(interior_wall, exterior_wall)
        self.wall = cv.add(self.wall, door)
        cv.imshow("wall", self.wall)
        # 生成墙体骨架和获取墙体厚度
        self.skeleton = self.get_binary_skeleton()
        cv.imshow("skeleton", self.skeleton)
        self.thickness, expansion = self.__calc_wall_thickness()
        # 输入墙体图生成网格图
        grid = Grid_img(self.wall, self.thickness)
        grid_img = grid.GenerateImage()
        cv.imshow("gird img", grid_img)
        # 生成气泡图
        bubble_img = self.generate_bubble_img()
        cv.imshow("bubble img", bubble_img)
        cv.waitKey(0)
        cv.destroyAllWindows()


if __name__ == '__main__':
    pass
    floor_plan = Floor_plan(cv.imread("./dataset/0.png"))
    floor_plan.getResult()
