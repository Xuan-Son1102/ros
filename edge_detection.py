#!/usr/bin/env python3
import cv2
import numpy as np
import rospy
from std_msgs.msg import Float32MultiArray

# === ROS NODE ===
rospy.init_node('lane_pid_controller')
pub_rpm = rospy.Publisher('wheel_rpm', Float32MultiArray, queue_size=10)

vidcap = cv2.VideoCapture("Video_Lane_demo/demo.mp4")
success, image = vidcap.read()

def nothing(x): pass

cv2.namedWindow("Trackbars")
cv2.createTrackbar("L - H","Trackbars", 101,255,nothing)
cv2.createTrackbar("L - S","Trackbars", 14,255,nothing)
cv2.createTrackbar("L - V","Trackbars", 0,255,nothing)
cv2.createTrackbar("U - H","Trackbars", 255,255,nothing)
cv2.createTrackbar("U - S","Trackbars", 255,255,nothing)
cv2.createTrackbar("U - V","Trackbars", 101,255,nothing)

prevLx, prevRx = [], []

# === Thông số robot ===
wheel_base = 34.0      # cm
wheel_radius = 5.0     # cm
robot_speed_max = 30.0 # cm/s

rate = rospy.Rate(20)  # ROS loop 20Hz

while not rospy.is_shutdown():
    success, image = vidcap.read()
    if not success:
        break
    frame = cv2.resize(image, (640,480))

    # === Perspective Transform ===
    tl, bl = (99, 165), (2, 453)
    tr, br = (485, 154), (636, 443)
    pts1 = np.float32([tl, bl, tr, br])
    pts2 = np.float32([[0,0], [0,480], [640,0], [640,480]])
    M = cv2.getPerspectiveTransform(pts1, pts2)
    birdseye = cv2.warpPerspective(frame, M, (640,480))

    hsv = cv2.cvtColor(birdseye, cv2.COLOR_BGR2HSV)

    # === Trackbar HSV ===
    l_h = cv2.getTrackbarPos("L - H","Trackbars")
    l_s = cv2.getTrackbarPos("L - S","Trackbars")
    l_v = cv2.getTrackbarPos("L - V","Trackbars")
    u_h = cv2.getTrackbarPos("U - H","Trackbars")
    u_s = cv2.getTrackbarPos("U - S","Trackbars")
    u_v = cv2.getTrackbarPos("U - V","Trackbars")

    lower, upper = np.array([l_h, l_s, l_v]), np.array([u_h, u_s, u_v])
    mask = cv2.inRange(hsv, lower, upper)

    # === Histogram tìm lane ===
    histogram = np.sum(mask[mask.shape[0]//2:, :], axis=0)
    midpoint = histogram.shape[0]//2
    left_base = np.argmax(histogram[:midpoint])
    right_base = np.argmax(histogram[midpoint:]) + midpoint

    # === Sliding Window ===
    y, lx, rx = 472, [], []
    msk = mask.copy()

    while y > 0:
        img = mask[y-40:y, left_base-50:left_base+50]
        contours, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            M = cv2.moments(c)
            if M["m00"] != 0:
                cx = int(M["m10"]/M["m00"])
                lx.append(left_base-50+cx)
                left_base = left_base-50+cx

        img = mask[y-40:y, right_base-50:right_base+50]
        contours, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            M = cv2.moments(c)
            if M["m00"] != 0:
                cx = int(M["m10"]/M["m00"])
                rx.append(right_base-50+cx)
                right_base = right_base-50+cx

        y -= 40

    if len(lx)==0: lx = prevLx
    else: prevLx = lx
    if len(rx)==0: rx = prevRx
    else: prevRx = rx

    min_len = min(len(lx), len(rx))
    if min_len>0:
        left_pts = [(lx[i], 472-i*40) for i in range(min_len)]
        right_pts = [(rx[i], 472-i*40) for i in range(min_len)]

        left_fit = np.polyfit([p[1] for p in left_pts], [p[0] for p in left_pts], 2)
        right_fit = np.polyfit([p[1] for p in right_pts], [p[0] for p in right_pts], 2)

        y_eval = 480
        left_curvature = ((1+(2*left_fit[0]*y_eval+left_fit[1])**2)**1.5)/abs(2*left_fit[0])
        right_curvature = ((1+(2*right_fit[0]*y_eval+right_fit[1])**2)**1.5)/abs(2*right_fit[0])
        curvature = (left_curvature+right_curvature)/2

        lane_center = (left_base+right_base)/2
        car_pos = 320
        lane_offset = (car_pos-lane_center)*3.7/640
        steering_angle = np.arctan(lane_offset/curvature)*180/np.pi

        # === Điều chỉnh tốc độ từng bánh theo hướng cua ===
        k_turn = min(0.5, abs(steering_angle) / 45)  # giảm tối đa 50%

        if steering_angle > 0:  # cua phải
            v_left = robot_speed_max * (1 - k_turn)
            v_right = robot_speed_max * (1 + k_turn * 0.3)
        elif steering_angle < 0:  # cua trái
            v_left = robot_speed_max * (1 + k_turn * 0.3)
            v_right = robot_speed_max * (1 - k_turn)
        else:  # đi thẳng
            v_left = v_right = robot_speed_max

        # === Chuyển sang RPM ===
        rpm_left = (v_left / wheel_radius) * 60 / (2 * np.pi)
        rpm_right = (v_right / wheel_radius) * 60 / (2 * np.pi)

        # === Gửi RPM qua ROS ===
        msg = Float32MultiArray()
        msg.data = [rpm_left, rpm_right]
        pub_rpm.publish(msg)

        # === Hiển thị ===
        cv2.putText(frame,f'Angle: {steering_angle:.1f}',(30,30),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)
        cv2.putText(frame,f'L RPM: {rpm_left:.1f}',(30,70),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,255),2)
        cv2.putText(frame,f'R RPM: {rpm_right:.1f}',(30,110),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,255),2)
        cv2.imshow("Lane Detection", frame)

    if cv2.waitKey(20) & 0xFF==27: break
    rate.sleep()

vidcap.release()
cv2.destroyAllWindows()
