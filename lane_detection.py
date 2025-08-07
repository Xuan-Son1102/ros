import cv2
import numpy as np
import serial
import time

# --- Cấu hình --- # <--- MỚI
# Cấu hình cổng Serial để giao tiếp với ESP32
SERIAL_PORT = 'COM4'  # THAY ĐỔI CỔNG COM CHO PHÙ HỢP
BAUDRATE = 115200

# Cấu hình robot và điều khiển
BASE_SPEED = 8
MAX_SPEED = 15
MOTOR_LEFT_ADDR = 1      
MOTOR_RIGHT_ADDR = 2   
FORWARD_DIR = 1          
REVERSE_DIR = 0       

# Cấu hình bộ điều khiển PID
KP = 0.45                # Gain tỷ lệ - Tinh chỉnh giá trị này đầu tiên
KI = 0.01                # Gain tích phân - Tăng từ từ nếu robot có sai số ổn định
KD = 0.05                # Gain vi phân - Tăng để giảm dao động


class PIDController:
    def __init__(self, Kp, Ki, Kd):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.prev_error = 0
        self.integral = 0

    def update(self, error, dt=1):
        self.integral += error * dt
        derivative = (error - self.prev_error) / dt
        self.prev_error = error
        return self.Kp * error + self.Ki * self.integral + self.Kd * derivative

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
4
cv2.namedWindow("Trackbars")

cv2.createTrackbar("L - H","Trackbars",  101,255,nothing)
cv2.createTrackbar("L - S","Trackbars",  14,255,nothing)
cv2.createTrackbar("L - V","Trackbars",0,255,nothing)
cv2.createTrackbar("U - H","Trackbars",255,255,nothing)
cv2.createTrackbar("U - S","Trackbars", 255,255,nothing)
cv2.createTrackbar("U - V","Trackbars",101,255,nothing)

pid = PIDController(KP, KI, KD) # <--- MỚI: Khởi tạo đối tượng PID
prevLx, prevRx = [], []

def nothing(x): pass

# --- Vòng lặp chính ---
while True:
    success, image = vidcap.read()
    if not success:
        vidcap.set(cv2.CAP_PROP_POS_FRAMES, 0) # Lặp lại video
        continue
        
    frame = cv2.resize(image, (640, 480))

    # (Phần xử lý ảnh và biến đổi perspective của bạn giữ nguyên)
    tl, bl = (56, 164), (2, 427)
    tr, br = (554, 180), (630, 429)
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
            img = mask[y-40:y, base-50:base+50]
            contours, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                Mmt = cv2.moments(cnt)
                if Mmt['m00'] != 0:
                    cx = int(Mmt['m10']/Mmt['m00'])
                    arr.append(base-50+cx)
                    if arr is lx: left_base = base-50+cx
                    else: right_base = base-50+cx
        y -= 40
    lx = lx if lx else prevLx
    rx = rx if rx else prevRx
    prevLx, prevRx = lx, rx

    if min(len(lx), len(rx)) > 0:
        # --- LOGIC ĐIỀU KHIỂN MỚI --- # <--- MỚI
        
        # 1. Tính toán sai số (độ lệch so với tâm)
        lane_center = (left_base + right_base) / 2
        robot_center = 320 # 640 / 2
        error = lane_center - robot_center

        # 2. Dùng PID để tính toán giá trị điều chỉnh
        correction = pid.update(error)

        # 3. Tính toán tốc độ cho mỗi bánh
        speed_left = BASE_SPEED - correction
        speed_right = BASE_SPEED + correction

        # 4. Giới hạn tốc độ trong khoảng cho phép
        speed_left = np.clip(speed_left, -MAX_SPEED, MAX_SPEED)
        speed_right = np.clip(speed_right, -MAX_SPEED, MAX_SPEED)
        
        # 5. Xác định chiều quay và tốc độ tuyệt đối để gửi lệnh
        dir_left = FORWARD_DIR if speed_left >= 0 else REVERSE_DIR
        val_left = abs(speed_left)

        dir_right = FORWARD_DIR if speed_right >= 0 else REVERSE_DIR
        val_right = abs(speed_right)

        # 6. Gửi lệnh tới ESP32
        send_motor_command(ser, MOTOR_LEFT_ADDR, dir_left, val_left)
        send_motor_command(ser, MOTOR_RIGHT_ADDR, dir_right, val_right)
        
        # --- Phần hiển thị (cập nhật để hiển thị tốc độ) ---
        # (Phần vẽ vời của bạn giữ nguyên)
        overlay = birdseye.copy()
        min_len = min(len(lx), len(rx))
        quad = np.array([(lx[0],472),(lx[min_len-1],0),(rx[min_len-1],0),(rx[0],472)], dtype=np.int32).reshape((-1,1,2))
        cv2.fillPoly(overlay,[quad],(0,150,0))
        cv2.addWeighted(overlay,0.2,birdseye,0.8,0,birdseye)
        invM = cv2.getPerspectiveTransform(pts2, pts1)
        warp_back = cv2.warpPerspective(birdseye, invM, (640,480))
        result = cv2.addWeighted(frame,1,warp_back,0.5,0)
        
        # Hiển thị thông tin điều khiển
        cv2.putText(result,f'Error: {error:.2f}',(30,30),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)
        cv2.putText(result,f'Correction: {correction:.2f}',(30,70),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)
        cv2.putText(result,f'L_Speed: {val_left:.1f} (Dir: {dir_left})',(30,110),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
        cv2.putText(result,f'R_Speed: {val_right:.1f} (Dir: {dir_right})',(30,150),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)

        cv2.imshow("Bird's Eye View", birdseye)
        cv2.imshow("Mask", mask)
        cv2.imshow('Result', result)
    else: # <--- MỚI: Dừng robot nếu không thấy làn đường
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
