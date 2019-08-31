import time
import numpy as np
import cv2
from matplotlib import pyplot as plt

FLANN_INDEX_KDTREE = 0
MIN_MATCH_COUNT = 10
MASK_IMG_NAME = 'img/idcard_mask.jpg'


def ocr(parse_img_name):
    pass


def find_idcard(parse_img_name):
    # imread 读取图片 格式为BGR  IMREAD_GRAYSCALE 以灰度读取一张图片
    mask_img = img_resize(cv2.UMat(cv2.imread(MASK_IMG_NAME, cv2.IMREAD_GRAYSCALE)), 640)

    parse_img = img_resize(cv2.UMat(cv2.imread(parse_img_name, cv2.IMREAD_GRAYSCALE)), 1920)
    img_org = img_resize(cv2.UMat(cv2.imread(parse_img_name)), 1920)

    sift = cv2.xfeatures2d.SIFT_create()
    # 特征点检测
    kp1, des1 = sift.detectAndCompute(mask_img, None)
    kp2, des2 = sift.detectAndCompute(parse_img, None)

    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=10)

    matches = cv2.FlannBasedMatcher(index_params, search_params).knnMatch(des1, des2, k=2)

    match__data = [m for m, n in matches if m.distance < 0.7 * n.distance]
    if len(match__data) < MIN_MATCH_COUNT:
        return

    src_pts = np.float32([kp1[m.queryIdx].pt for m in match__data]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in match__data]).reshape(-1, 1, 2)
    m, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    h, w = cv2.UMat.get(mask_img).shape
    m_r = np.linalg.inv(m)
    result_img = cv2.warpPerspective(img_org, m_r, (w, h))

    # 特征点匹配结果
    # draw_params = dict(matchColor=(0, 255, 0),
    #                    singlePointColor=None,
    #                    matchesMask=mask.ravel().tolist(),
    #                    flags=2)
    # img3 = cv2.drawMatches(mask_img, kp1, parse_img, kp2, match__data, None, **draw_params)
    # show_img(img3)
    # plt.show()

    return result_img


def img_resize(imggray, dwidth):
    crop = imggray
    size = crop.get().shape
    height = size[0]
    width = size[1]
    height = height * dwidth / width
    crop = cv2.resize(src=crop, dsize=(dwidth, int(height)), interpolation=cv2.INTER_CUBIC)
    
    return crop


def show_img(img):
    cv2.namedWindow("contours", 0)
    cv2.resizeWindow("contours", 1600, 1200)
    cv2.imshow("contours", img)
    cv2.waitKey()
