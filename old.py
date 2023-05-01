# python-38
# ------------------------------------------------
# 作者     :刘想
# 时间     :2023/2/18 15:51
# ------------------------------------------------

def i0_3x3(mat, i, j):
    p1 = mat[i - 1, j - 1]
    p2 = mat[i, j - 1]
    p3 = mat[i + 1, j - 1]
    p4 = mat[i - 1, j]
    p5 = mat[i, j]
    p6 = mat[i + 1, j]
    p7 = mat[i - 1, j + 1]
    p8 = mat[i, j + 1]
    p9 = mat[i + 1, j + 1]
    if p5 == p6 and p6 == p8 and p8 == p9 and p1 != p5 and p2 != p5 and p3 != p5 and p4 != p5 and p7 != p5:
        return True
    elif p4 == p5 and p5 == p7 and p7 == p8 and p1 != p5 and p2 != p5 and p3 != p5 and p6 != p5 and p9 != p5:
        return True
    elif p1 == p2 and p2 == p5 and p5 == p4 and p3 != p5 and p6 != p5 and p7 != p5 and p8 != p5 and p9 != p5:
        return True
    elif p2 == p3 and p3 == p6 and p6 == p5 and p1 != p5 and p4 != p5 and p7 != p5 and p8 != p5 and p9 != p5:
        return True
    elif p1 == p2 and p2 == p3 and p3 == p4 and p4 == p5 and p5 == p6 and p7 != p5 and p8 != p5 and p9 != p5:
        return True
    elif p2 == p5 and p5 == p8 and p8 == p9 and p9 == p6 and p6 == p3 and p1 != p5 and p4 != p5 and p7 != p5:
        return True
    elif p4 == p5 and p5 == p6 and p6 == p7 and p7 == p8 and p8 == p9 and p1 != p5 and p2 != p5 and p3 != p5:
        return True
    elif p1 == p4 and p4 == p7 and p7 == p8 and p8 == p5 and p5 == p2 and p3 != p5 and p6 != p5 and p9 != p5:
        return True
    else:
        return False


def get_gray_map(gray):
    # 根据灰度图计算每一个像素是否具有平直度, 若该像素具有平直度则加入一个字典中进行统计 类似于直方图
    gray_map = dict()
    sum_map = dict()
    w, h = gray.shape
    for i in range(1, w - 1):
        for j in range(1, h - 1):
            if gray[i, j] in sum_map.keys():
                sum_map[gray[i, j]] = sum_map[gray[i, j]] + 1
            else:
                sum_map[gray[i, j]] = 1
            if i0_3x3(gray, i, j):
                if gray[i, j] in gray_map.keys():
                    gray_map[gray[i, j]] = gray_map[gray[i, j]] + 1
                else:
                    gray_map[gray[i, j]] = 1
    return gray_map, sum_map


def get_wall(img):
    # 灰度
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    cv.imshow("gray", gray)
    # 直方图均衡 增强
    cla_he = cv.createCLAHE(clipLimit=5, tileGridSize=(3, 3))
    enhance = cla_he.apply(gray)
    cv.imshow("Image2", enhance)
    # 计算各像素平直度 和 像素面积
    gray_map, sum_map = get_gray_map(gray)
    print(gray_map)
    print(sum_map)

    pixels = [1.0 * gray_map[i] / sum_map[i] for i in gray_map.keys()]
    wall_pixels = list()
    # 1.0 * sum(wall_pixels) / len(wall_pixels) + 0.1
    for i in gray_map.keys():
        if 1.0 * gray_map[i] / sum_map[i] > sum(pixels) / len(pixels):
            wall_pixels.append(i)
        print(i, ": ", 1.0 * gray_map[i] / sum_map[i])
    binary = gray.copy()
    w, h = gray.shape
    for i in range(0, w):
        for j in range(0, h):
            if gray[i, j] in wall_pixels:
                binary[i, j] = 255
            else:
                binary[i, j] = 0
    cv.imshow("wall", binary)
    # wall = cv.adaptiveThreshold(img2, img2.max(), cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, -30)
    # cv.imshow("wall", wall)
    # hist = cv.calcHist([img2], [0], None, [256], [0, 255])
    # plt.hist(gray.ravel(), 256, [0, 256])
    # plt.show()
    # print(hist)
    result = cv.connectedComponents(enhance, connectivity=8, ltype=cv.CV_32S)
    # color = [[0, 0, 0], [255, 255, 255]]
    labels = result[1]
    shape = list(labels.shape)
    shape.append(3)
    # labels_img = np.zeros(shape, np.uint8)
    # for i in range(0, shape[0]):
    #     for j in range(0, shape[1]):
    #         labels_img[i, j] = color[labels[i, j]]

    # labels = labels * 255
    # cv.imshow("labels", labels_img)
