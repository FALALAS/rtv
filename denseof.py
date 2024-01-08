import cv2
import numpy as np

cap = cv2.VideoCapture('output_video000.mp4')

ret, frame1 = cap.read()
frame1_gray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
hsv = np.zeros_like(frame1)

hsv[..., 1] = 255

# 定义视频编码器和创建 VideoWriter 对象
video_name = 'disof.mp4'
fps = 30
height, width, layers = frame1.shape
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video = cv2.VideoWriter(video_name, fourcc, fps, (width, height))

while (1):
    ret, frame2 = cap.read()
    if ret:
        frame2_gray = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        #flow = cv2.calcOpticalFlowFarneback(frame1_gray, frame2_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        dis = cv2.DISOpticalFlow_create(0)
        flow = dis.calc(frame1_gray, frame2_gray, None, )
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        video.write(bgr)
        cv2.imshow('frame2', bgr)
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            cv2.destroyAllWindows()
            break
        elif k == ord('s'):
            cv2.imwrite('opticalfb.png', frame2)
            cv2.imwrite('opticalhsv.png', bgr)
        frame1_gray = frame2_gray
    else:
        cv2.destroyAllWindows()
        break
