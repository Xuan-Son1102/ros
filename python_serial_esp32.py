import cv2
import numpy as np
import serial
import time

# --- Cấu hình ---
SERIAL_PORT = 'COM4'  # THAY ĐỔI CỔNG COM CHO PHÙ HỢP
BAUDRATE = 115200

# Cấu hình robot và điều khiển
BASE_SPEED = 15          
MAX_SPEED = 20      
MOTOR_LEFT_ADDR = 1      
MOTOR_RIGHT_ADDR = 2   
FORWARD_DIR = 0          
REVERSE_DIR = 1       

# Cấu hình bộ điều khiển PID
KP = 0.35                # Gain tỷ lệ - Tinh chỉnh giá trị này đầu tiên
KI = 0.01                # Gain tích phân - Tăng từ từ nếu robot có sai số ổn định
KD = 0.05                # Gain vi phân - Tăng để giảm dao động


class PIDController:
    def __init__(self, Kp, Ki, Kd):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.prev_error = 0.0
        self.integral = 0.0

    def update(self, error, dt=1.0):
        if dt <= 0:
            dt = 1.0
        self.integral += error * dt
        derivative = (error - self.prev_error) / dt
        self.prev_error = error
        return self.Kp * error + self.Ki * self.integral + self.Kd * derivative


class KalmanFilter1D:
    """
    Đơn giản 1D Kalman filter (random walk process model).
    state x, estimation error covariance p
    Q: process variance, R: measurement variance
    """
    def __init__(self, q=1.0, r=10.0, x0=0.0, p0=1.0):
        self.q = q
        self.r = r
        self.x = x0
        self.p = p0

    def set_process_measurement(self, q, r):
        self.q = q
        self.r = r

    def update(self, z):
        # Predict (for 1D random-walk, prediction step just increases uncertainty)
        self.p = self.p + self.q
        # Kalman gain
        k = self.p / (self.p + self.r)
        # Update estimate with measurement z
        self.x = self.x + k * (z - self.x)
        # Update covariance
        self.p = (1 - k) * self.p
        return self.x


def send_motor_command(ser, address, direction, speed):

    if not ser or not ser.is_open:
        # print("Serial port is not open.")
        return

    # Giới hạn tốc độ trong khoảng 0-255
    speed = int(np.clip(speed, 0, 255))
    
    # Xây dựng Byte 1: (bit hướng << 7) | địa chỉ
    byte1 = (direction << 7) | (address & 0x7F)
    
    # Byte 2: Tốc độ
    byte2 = speed
    
    # Byte 3: Byte kết thúc
    byte3 = 0xFF
    
    command = bytes([byte1, byte2, byte3])
    ser.write(command)


try:
    ser = serial.Serial(SERIAL_PORT, BAUDRATE, timeout=1)
    time.sleep(2)
    print(f"Connected to ESP32 on {SERIAL_PORT} at {BAUDRATE} baud.")
except serial.SerialException as e:
    print(f"Error opening serial port: {e}")
    ser = None

# --- Khởi tạo các thành phần ---
vidcap = cv2.VideoCapture(0)

def nothing(x):
    pass

cv2.namedWindow("Trackbars")

# Các trackbar cho màu
cv2.createTrackbar("L - H","Trackbars",  101,255,nothing)
cv2.createTrackbar("L - S","Trackbars",  14,255,nothing)
cv2.createTrackbar("L - V","Trackbars",0,255,nothing)
cv2.createTrackbar("U - H","Trackbars",255,255,nothing)
cv2.createTrackbar("U - S","Trackbars", 255,255,nothing)
cv2.createTrackbar("U - V","Trackbars",101,255,nothing)

# Trackbar để tinh chỉnh Kalman filter (giá trị sẽ được chia cho 1000 để lấy float)
cv2.createTrackbar("Kal_Q","Trackbars",1,1000,nothing)   # process variance (q)
cv2.createTrackbar("Kal_R","Trackbars",50,1000,nothing)  # measurement variance (r)

pid = PIDController(KP, KI, KD) # Khởi tạo đối tượng PID
prevLx, prevRx = [], []

# Khởi tạo Kalman
kalman = KalmanFilter1D(q=0.001, r=0.05, x0=0.0, p0=1.0)

last_time = time.time()

# --- Vòng lặp chính ---
while True:
    success, image = vidcap.read()
    if not success:
        vidcap.set(cv2.CAP_PROP_POS_FRAMES, 0) # Lặp lại video
        continue
        
    frame = cv2.resize(image, (640, 480))

    # (Phần xử lý ảnh và biến đổi perspective của bạn giữ nguyên)
    tl, bl = (13, 185), (4, 422)
    tr, br = (596, 204), (628, 410)
    pts1 = np.float32([tl, bl, tr, br])
    pts2 = np.float32([[0,0], [0,480], [640,0], [640,480]])
    M = cv2.getPerspectiveTransform(pts1, pts2)
    birdseye = cv2.warpPerspective(frame, M, (640,480))

    hsv = cv2.cvtColor(birdseye, cv2.COLOR_BGR2HSV)
    l_h, l_s, l_v = [cv2.getTrackbarPos(t,"Trackbars") for t in ("L - H","L - S","L - V")]
    u_h, u_s, u_v = [cv2.getTrackbarPos(t,"Trackbars") for t in ("U - H","U - S","U - V")]
    mask = cv2.inRange(hsv, np.array([l_h, l_s, l_v]), np.array([u_h, u_s, u_v]))

    histogram = np.sum(mask[mask.shape[0]//2:, :], axis=0)
    midpoint = histogram.shape[0]//2
    left_base = np.argmax(histogram[:midpoint])
    right_base = np.argmax(histogram[midpoint:]) + midpoint

    # (Phần dò làn của bạn giữ nguyên)
    y, lx, rx = 472, [], []
    while y > 0:
        for base, arr in [(left_base, lx), (right_base, rx)]:
            # bảo đảm index hợp lệ
            x1 = max(0, base-50)
            x2 = min(mask.shape[1], base+50)
            y1 = max(0, y-40)
            y2 = min(mask.shape[0], y)
            img = mask[y1:y2, x1:x2]
            if img.size == 0:
                continue
            contours, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                Mmt = cv2.moments(cnt)
                if Mmt['m00'] != 0:
                    cx = int(Mmt['m10']/Mmt['m00'])
                    arr.append(x1+cx)
                    if arr is lx: left_base = x1+cx
                    else: right_base = x1+cx
        y -= 40
    lx = lx if lx else prevLx
    rx = rx if rx else prevRx
    prevLx, prevRx = lx, rx

    if min(len(lx), len(rx)) > 0:
        # --- LOGIC ĐIỀU KHIỂN MỚI (với Kalman cho error) ---

        # tính dt thực tế cho PID
        now = time.time()
        dt = now - last_time
        last_time = now
        if dt <= 0:
            dt = 0.01

        # 1. Tính toán sai số (độ lệch so với tâm)
        lane_center = (left_base + right_base) / 2.0
        robot_center = 320.0 # 640 / 2
        raw_error = lane_center - robot_center

        # 2. Cập nhật tham số Kalman từ trackbar (cho phép tune realtime)
        q_tb = max(1, cv2.getTrackbarPos("Kal_Q","Trackbars"))
        r_tb = max(1, cv2.getTrackbarPos("Kal_R","Trackbars"))
        # chuyển sang float nhỏ hơn
        q = q_tb / 1000.0
        r = r_tb / 1000.0
        kalman.set_process_measurement(q, r)

        # 3. Lọc sai số bằng Kalman
        filtered_error = kalman.update(raw_error)

        # 4. Dùng PID để tính toán giá trị điều chỉnh (dùng filtered_error thay vì raw)
        correction = pid.update(filtered_error, dt=dt)

        # 5. Tính toán tốc độ cho mỗi bánh
        speed_left = BASE_SPEED - correction
        speed_right = BASE_SPEED + correction

        # 6. Giới hạn tốc độ trong khoảng cho phép
        speed_left = np.clip(speed_left, -MAX_SPEED, MAX_SPEED)
        speed_right = np.clip(speed_right, -MAX_SPEED, MAX_SPEED)
        
        # 7. Xác định chiều quay và tốc độ tuyệt đối để gửi lệnh
        dir_left = FORWARD_DIR if speed_left >= 0 else REVERSE_DIR
        val_left = abs(speed_left)

        dir_right = FORWARD_DIR if speed_right >= 0 else REVERSE_DIR
        val_right = abs(speed_right)

        # 8. Gửi lệnh tới ESP32
        send_motor_command(ser, MOTOR_LEFT_ADDR, dir_left, val_left)
        send_motor_command(ser, MOTOR_RIGHT_ADDR, dir_right, val_right)
        
        # --- Phần hiển thị (cập nhật để hiển thị tốc độ) ---
        overlay = birdseye.copy()
        min_len = min(len(lx), len(rx))
        quad = np.array([(lx[0],472),(lx[min_len-1],0),(rx[min_len-1],0),(rx[0],472)], dtype=np.int32).reshape((-1,1,2))
        cv2.fillPoly(overlay,[quad],(0,150,0))
        cv2.addWeighted(overlay,0.2,birdseye,0.8,0,birdseye)
        invM = cv2.getPerspectiveTransform(pts2, pts1)
        warp_back = cv2.warpPerspective(birdseye, invM, (640,480))
        result = cv2.addWeighted(frame,1,warp_back,0.5,0)
        
        # Hiển thị thông tin điều khiển
        cv2.putText(result,f'RawErr: {raw_error:.2f}',(30,30),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)
        cv2.putText(result,f'FiltErr: {filtered_error:.2f}',(30,60),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)
        cv2.putText(result,f'Correction: {correction:.2f}',(30,95),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)
        cv2.putText(result,f'L_Speed: {val_left:.1f} (Dir: {dir_left})',(30,130),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,255,0),2)
        cv2.putText(result,f'R_Speed: {val_right:.1f} (Dir: {dir_right})',(30,165),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,255,0),2)
        cv2.putText(result,f'Kal_Q: {q:.3f} R: {r:.3f}',(30,200),cv2.FONT_HERSHEY_SIMPLEX,0.6,(200,200,0),2)

        cv2.imshow("Bird's Eye View", birdseye)
        cv2.imshow("Mask", mask)
        cv2.imshow('Result', result)
    else: # Dừng robot nếu không thấy làn đường
        send_motor_command(ser, MOTOR_LEFT_ADDR, FORWARD_DIR, 0)
        send_motor_command(ser, MOTOR_RIGHT_ADDR, FORWARD_DIR, 0)


    if cv2.waitKey(20) & 0xFF == 27: # Thay đổi waitKey để video chạy mượt hơn
        # Dừng hẳn robot trước khi thoát
        send_motor_command(ser, MOTOR_LEFT_ADDR, FORWARD_DIR, 0)
        send_motor_command(ser, MOTOR_RIGHT_ADDR, FORWARD_DIR, 0)
        break

vidcap.release()
if ser and ser.is_open:
    ser.close()
cv2.destroyAllWindows()
