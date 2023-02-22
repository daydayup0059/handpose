# https://mp.weixin.qq.com/s/o_gfooaBHGy6Cr1ZqdvkTg
#通过手势识别，显示几个手指，并在图片上实时出现 检测出的数字
# “sys”是“system”的缩写
import sys
import numpy as np
import cv2
import math

# cv2.FONT_HERSHEY_SIMPLEX 设置字体类型(正常大小的sans-serif字体)
font = cv2.FONT_HERSHEY_SIMPLEX


# cv2.contourArea(cnt， True)  # 计算轮廓的面积，参数说明：cnt为输入的单个轮廓值
def cnt_area(cnt):
    area = cv2.contourArea(cnt)
    return area


# HSV提取肤色轮廓，然后筛选找出手部轮廓
def Gesture_Recognize(img):
    #定义检测出的手指数pointNum
    pointNum = 1
    # cv2.COLOR_BGR2HSV是rgb转hsv的函数
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_hsv_1 = np.array([0, 50, 50])  # 颜色范围低阈值
    upper_hsv_1 = np.array([20, 255, 255])  # 颜色范围高阈值
    lower_hsv_2 = np.array([150, 50, 50])  # 颜色范围低阈值
    upper_hsv_2 = np.array([180, 255, 255])  # 颜色范围高阈值
    # mask = cv2.inRange(hsv, lower_red, upper_red) #lower20===>0,upper200==>0,，lower～upper==>255
    # cv2.inRange函数很简单，参数有三个
    # 第一个参数：hsv指的是原图
    # 第二个参数：lower_red指的是图像中低于这个lower_red的值，图像值变为0
    # 第三个参数：upper_red指的是图像中高于这个upper_red的值，图像值变为0
    # 而在lower_red～upper_red之间的值变成255
    mask1 = cv2.inRange(hsv_img, lower_hsv_1, upper_hsv_1)
    mask2 = cv2.inRange(hsv_img, lower_hsv_2, upper_hsv_2)
    # 因为2个mask检测出的内容可能不在同一个位置，所以图片相加，合在同一张图片上，不能用从cv.add
    mask = mask1 + mask2
    # 中值滤波，去除一些边缘噪点
    mask = cv2.medianBlur(mask, 5)
    # 返回(5,5)单位矩阵
    k1 = np.ones((5, 5), np.uint8)
    #闭运算，先膨胀，后腐蚀
    # 膨胀
    mask = cv2.dilate(mask, k1, iterations=1)
    # 腐蚀
    mask = cv2.erode(mask, k1, iterations=1)
    # 灰度图转换为BGR
    mask_color = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    #mask是黑白图像,
    cv2.imshow("mask", mask)
    # cv2.imwrite("mask.png", mask)
    # 生产0矩阵，形状和mask形状相同
    black_img = np.zeros(mask.shape, np.uint8)
    # cv2.findContours查找轮廓https://blog.csdn.net/vclearner2/article/details/120776685
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # def return语句可以返回多个值，以逗号分隔，实际返回的是一个tuple。
    if len(contours) < 1:
        return 0, img
    # 排序reverse：True表示降序 False表示降序
    # .sort我们必须使用列表而不是元组，所以先把 contours转换为列表
    contours = list(contours)
    #key=cnt_area,调用方法计算轮廓面积
    #list.sort 用法 https://www.runoob.com/python3/python3-att-list-sort.html
    #sort(key=cnt_area取自于可迭代对象中，指定可迭代对象中的一个元素来进行排序。
    contours.sort(key=cnt_area, reverse=True)
    #sort排序的意义取本次检测图片的最大轮廓，reverse=True表示降序，contours[0]是最大轮廓
    # img是一个二值图，也就是它的参数；返回四个值，分别是x，y，w，h；x，y是矩阵左上点的坐标，w，h是矩阵的宽和高
    #cv2.boundingRect用一个最小的矩形，把找到的形状包起来
    (x0, y0, w0, h0) = cv2.boundingRect(contours[0])

    if (w0 >= 100 and h0 >= 100):
        cv2.rectangle(img, (x0, y0), (x0 + w0, y0 + h0), (255, 0, 255), 2)
        # cv2.arcLength(cnt, True) 计算轮廓的周长 参数2 表示轮廓是否封闭
        epsilon = 0.01 * cv2.arcLength(contours[0], True)
        # 主要功能是把一个连续光滑曲线折线化
        approx = cv2.approxPolyDP(contours[0], epsilon, True)
        # cv2.drawContours()函数的功能是绘制轮廓
        #下文[approx]，必须要加[],[]为list列表数据类型
        #approx=list(approx)
        #cv2.drawContours(black_img, approx, -1, (255, 0, 255), 2)
        print('approx.shape:',approx.shape)   #(11, 1, 2)
        print('type(approx):', type(approx))  # numpy.ndarray  然后，ndarray本质是数组数组里面嵌套数组

        #[approx]: 检测出的轮廓。 在全黑图片上画轮廓
        cv2.drawContours(black_img, [approx], -1, (255, 0, 255), 2)

        cv2.imshow("black_img", black_img)
        contours2, hierarchy2 = cv2.findContours(black_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) < 1:
            return 0, img
            # 寻找凸包，得到凸包的角点,hull：返回值，为凸包角点。可以理解为多边形的点坐标，或索引，hull输出凸包结果，n * 1 *2 数据结构，n为外包围圈点数
        hull = cv2.convexHull(contours2[0], returnPoints=False)
        # cv2.convexityDefects()计算凸包缺陷，图像上以 红色的点表示 ，时,returnPoints 需为 False
        #convexityDefects：返回值，为凸缺陷点集。它是一个数组，返回的指包括[起点，终点，轮廓上的距离凸包最远点，最远点到凸包的近似距离]
        defects = cv2.convexityDefects(contours2[0], hull)
        print('defects.shape:', defects.shape) #(12, 1, 4)
        print('type(defects):', type(defects))  #<class 'numpy.ndarray'>
        if defects is None:
            # print ('have no convex defects')
            pass
        else:
            # print ('have convex defects')
            for i in range(0, defects.shape[0]):
                #
                s, e, f, d = defects[i, 0]
                pre_start = (0, 0)
                pre_end = (0, 0)
                #起点
                start = tuple(contours2[0][s][0])
                #终点
                end = tuple(contours2[0][e][0])
                #轮廓上的距离凸包最远点
                far = tuple(contours2[0][f][0])
                # print(d)
                if d >= 13000:
                    #BGR格式
                    cv2.line(img, start, end, [0, 255, 0], 3)  # 凸包-绿色
                    cv2.circle(img, start, 10, [0, 255, 255], 3)  #起点画圆-绿色
                    cv2.circle(img, end, 10, [0, 255, 255], 3)  #终点画圆-红色
                    cv2.circle(img, far, 10, [0, 0, 255], 3)  # 凸包缺陷点-蓝色
                    pre_start = start
                    pre_end = end
                    pointNum += 1
    # 在图像上绘制文本，pointNum  为 检测出来的手指数
    #'hand-%d' % pointNum   输出为 hand-pointNum，hand-1，hand-2
    cv2.putText(img, 'hand-%d' % pointNum, (10, 35), font, 1.2, (0, 255, 255), 3)
    return pointNum, img


if __name__ == '__main__':
    # cv2.VideoCapture视频处理
    cap = cv2.VideoCapture()

    flag = cap.open(0)
    if flag:
        while True:
            # cap.read()获取视频中的每一帧图像
            ret, frame = cap.read()
            if frame is None:
                break
            cv2.imshow("frame", frame)
            # 调用上面自定义的Gesture_Recognize方法
            num, img = Gesture_Recognize(frame)
            cv2.imshow("Gesture_Recognize", img)
            char = cv2.waitKey(10)
            if char == 27:
                break
        # https://blog.csdn.net/qq_45712772/article/details/109250148关闭窗口
        cv2.destroyAllWindows()
        cap.release()


