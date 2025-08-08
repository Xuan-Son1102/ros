import cv2
import numpy as np
import math
import serial
import time

# --- Cấu hình Serial để giao tiếp ESP32 ---
SERIAL_PORT    = 'COM11'    # THAY ĐỔI CHO PHÙ HỢP
BAUDRATE       = 115200
SERIAL_TIMEOUT = 1          # giây

# --- Cấu hình động cơ ---
MAX_SPEED_CMD = 20         # giá trị max gửi qua Serial
WHEELBASE     = 0.42        # mét, dùng nếu có PID steering

def send_motor_command(ser, address, direction, speed):
    """
    Gửi gói 3 byte: [dir<<7 | addr, speed (0–255), 0xFF]
    direction: 1 = tiến, 0 = lùi
    """
    if ser is None or not ser.is_open:
        return
    b1 = (direction << 7) | (address & 0x7F)
    b2 = int(np.clip(speed, 0, MAX_SPEED_CMD))
    packet = bytes([b1, b2, 0xFF])
    ser.write(packet)

# --- Hàm xử lý lane (giữ nguyên của bạn) ---
def find_lane_lines(img):
    gray      = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_gauss = cv2.GaussianBlur(gray, (11, 11), 0)
    return cv2.Canny(img_gauss, 150, 200)

def birdview_transform(img):
    h, w = img.shape[:2]
    src = np.float32([[0, h], [w, h], [0, h*0.4], [w, h*0.4]])
    dst = np.float32([[240, h], [w-240, h], [-160, 0], [w+160, 0]])
    M = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(img, M, (w, h))

def find_left_right_points(image, draw=None):
    h, w = image.shape[:2]
    y = int(h * 0.9)
    if draw is not None:
        cv2.line(draw, (0, y), (w, y), (0,0,255), 2)
    scan = image[y, :]
    center = w//2
    left_pt  = next((x for x in range(center, 0, -1) if scan[x]>0), -1)
    right_pt = next((x for x in range(center+1, w) if scan[x]>0), -1)
    lane_w = 100
    if left_pt!=-1 and right_pt==-1:
        right_pt = left_pt + lane_w
    if right_pt!=-1 and left_pt==-1:
        left_pt = right_pt - lane_w
    if draw is not None:
        if left_pt!=-1:
            cv2.circle(draw, (left_pt, y), 7, (255,255,0), -1)
        if right_pt!=-1:
            cv2.circle(draw, (right_pt, y), 7, (0,255,0), -1)
    return left_pt, right_pt

def calculate_control_signal(frame):
    img_lines = find_lane_lines(frame)
    bird      = birdview_transform(img_lines)
    overlay   = birdview_transform(frame.copy())
    left, right = find_left_right_points(bird, draw=overlay)

    throttle = 0.5
    steering = 0.0
    im_c = frame.shape[1]//2
    if left!=-1 and right!=-1:
        center_lane = (left + right)//2
        diff = im_c - center_lane
        steering = -0.01 * diff

    return throttle, steering, overlay

def calculate_wheel_speeds(steering, throttle, wheelbase=WHEELBASE, max_speed=10):
    v_center = throttle * max_speed
    if abs(steering) < 1e-5:
        return v_center, v_center
    R = wheelbase / abs(math.tan(steering))
    if steering > 0:  # quay phải
        vL = v_center * (R + wheelbase/2) / R
        vR = v_center * (R - wheelbase/2) / R
    else:             # quay trái
        vL = v_center * (R - wheelbase/2) / R
        vR = v_center * (R + wheelbase/2) / R
    return vL, vR

# --- Main với Serial ---
def main():
    # Khởi tạo serial
    try:
        ser = serial.Serial(SERIAL_PORT, BAUDRATE, timeout=SERIAL_TIMEOUT)
        time.sleep(2)
        print(f"[INFO] Kết nối ESP32 trên {SERIAL_PORT} @ {BAUDRATE}")
    except Exception as e:
        print(f"[ERROR] Mở cổng Serial thất bại: {e}")
        ser = None

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Không mở được video/camera")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        throttle, steering, annotated = calculate_control_signal(frame)
        vL, vR = calculate_wheel_speeds(steering, throttle)

        # Map speed xuống 0–255 và xác định direction
        addrL, addrR = 1, 2
        dirL = 1 if vL>=0 else 0
        dirR = 1 if vR>=0 else 0
        cmdL = abs(vL)/throttle * MAX_SPEED_CMD if throttle>0 else 0
        cmdR = abs(vR)/throttle * MAX_SPEED_CMD if throttle>0 else 0

        send_motor_command(ser, addrL, dirL, cmdL)
        send_motor_command(ser, addrR, dirR, cmdR)

        # Hiển thị lên cửa sổ
        cv2.putText(annotated, f"Throt: {throttle:.2f}", (10,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255),2)
        cv2.putText(annotated, f"Steer: {steering:.2f}", (10,60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255),2)
        cv2.putText(annotated, f"vL: {vL:.2f}", (10,90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0),2)
        cv2.putText(annotated, f"vR: {vR:.2f}", (10,120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0),2)

        cv2.imshow("Lane Detection", annotated)

        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    # Dọn dẹp
    cap.release()
    if ser and ser.is_open:
        ser.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
