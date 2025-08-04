import cv2
import numpy as np

# ✅ Hàm phát hiện vạch kẻ bằng Canny
def find_lane_lines(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_gauss = cv2.GaussianBlur(gray, (11, 11), 0)
    thresh_low = 110
    thresh_high = 205
    img_canny = cv2.Canny(img_gauss, thresh_low, thresh_high)
    return img_canny

# ✅ Hàm biến đổi phối cảnh
def birdview_transform(img):
    IMAGE_H, IMAGE_W = img.shape[:2]
    src = np.float32([
        [0, IMAGE_H],
        [IMAGE_W, IMAGE_H],
        [0, IMAGE_H // 3],
        [IMAGE_W, IMAGE_H // 3]
    ])
    dst = np.float32([
        [90, IMAGE_H],
        [230, IMAGE_H],
        [-10, 0],
        [IMAGE_W + 10, 0]
    ])
    M = cv2.getPerspectiveTransform(src, dst)
    warped_img = cv2.warpPerspective(img, M, (IMAGE_W, IMAGE_H))
    return warped_img

# ✅ Hàm tìm điểm trái/phải
def find_left_right_points(image, draw=False):
    im_height, im_width = image.shape[:2]
    if draw: viz_img = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    interested_line_y = int(im_height * 0.7)
    if draw: cv2.line(viz_img, (0, interested_line_y), (im_width, interested_line_y), (0, 0, 255), 2)
    interested_line = image[interested_line_y, :]

    left_point = -1
    right_point = -1
    lane_width = 100
    center = im_width // 2

    for x in range(center, 0, -1):
        if interested_line[x] > 0:
            left_point = x
            break
    for x in range(center + 1, im_width):
        if interested_line[x] > 0:
            right_point = x
            break

    if left_point != -1 and right_point == -1:
        right_point = left_point + lane_width
    if right_point != -1 and left_point == -1:
        left_point = right_point - lane_width

    if draw:
        if left_point != -1:
            viz_img = cv2.circle(viz_img, (left_point, interested_line_y), 7, (255, 255, 0), -1)
        if right_point != -1:
            viz_img = cv2.circle(viz_img, (right_point, interested_line_y), 7, (0, 255, 0), -1)

    return left_point, right_point, viz_img

# ✅ Tính góc lái & tốc độ
def calculate_control_signal(frame):
    img = cv2.resize(frame, (320, 160))  # Resize cho nhanh
    img_lines = find_lane_lines(img)
    img_birdview = birdview_transform(img_lines)
    left_point, right_point, viz_img = find_left_right_points(img_birdview, draw=True)

    throttle = 0.5
    steering_angle = 0
    im_center = img.shape[1] // 2

    if left_point != -1 and right_point != -1:
        center_point = (right_point + left_point) // 2
        center_diff = im_center - center_point
        steering_angle = -float(center_diff * 0.01)

    return throttle, steering_angle, viz_img

# ✅ Mở webcam/video
cap = cv2.VideoCapture('Video_Lane_demo/demo.mp4')  # Dùng webcam, hoặc đổi "0" thành đường dẫn video

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    img = cv2.resize(frame, (320, 160))  # Resize cho nhanh
    img_lines = find_lane_lines(img)
    img_birdview = birdview_transform(img_lines)
    left_right_points = find_left_right_points(img_birdview, draw=True)
    cv2.imshow("Lane Detection", left_right_points[2])

   

    if cv2.waitKey(30) & 0xFF == 27:  # Nhấn ESC để thoát
        break

cap.release()
cv2.destroyAllWindows()
