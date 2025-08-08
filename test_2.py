import cv2
import numpy as np
import math
import serial
import time

# =============================================================================
# PHẦN 1: CẤU HÌNH THAM SỐ
# =============================================================================
VIDEO_PATH = 0    # Đường dẫn video đầu vào
DESIRED_SPEED = 1.0                # Vận tốc dài mong muốn của robot (m/s)
WHEEL_BASE = 0.42                  # Khoảng cách giữa hai bánh (m)
CAMERA_FOV_DEG = 60                # Góc nhìn ngang của camera (độ)
FRAME_WIDTH = 320                  # Chiều rộng ảnh sau resize
FRAME_HEIGHT = 160                 # Chiều cao ảnh sau resize

# --- Cấu hình cổng Serial để giao tiếp với ESP32 ---
SERIAL_PORT = 'COM11'  # THAY ĐỔI CỔNG COM CHO PHÙ HỢP
BAUDRATE = 115200

# --- Cấu hình Robot và Điều khiển (PWM 0-255) ---
BASE_SPEED_PWM = 10     # Tốc độ PWM tương ứng với DESIRED_SPEED khi đi thẳng
MAX_SPEED_PWM = 18      # Tốc độ PWM tối đa cho phép của mỗi bánh
MOTOR_LEFT_ADDR = 1       # Địa chỉ động cơ trái
MOTOR_RIGHT_ADDR = 2      # Địa chỉ động cơ phải
FORWARD_DIR = 1           # Hướng đi tới (sẽ là hướng duy nhất được dùng)

# =============================================================================
# PHẦN 2: KHỞI TẠO GIAO TIẾP SERIAL
# =============================================================================
ser = None
try:
    ser = serial.Serial(SERIAL_PORT, BAUDRATE, timeout=1)
    time.sleep(2)
    print(f"✅ Đã kết nối thành công tới ESP32 trên cổng {SERIAL_PORT}.")
except serial.SerialException as e:
    print(f"❌ Lỗi: Không thể mở cổng serial {SERIAL_PORT}. Chạy ở chế độ mô phỏng.")

def send_motor_command(ser, address, direction, speed):
    """Hàm gửi lệnh điều khiển tới một động cơ qua cổng Serial."""
    if not ser or not ser.is_open:
        return
    speed_val = int(speed)
    byte1 = (direction << 7) | (address & 0x7F)
    byte2 = speed_val
    byte3 = 0xFF
    command = bytes([byte1, byte2, byte3])
    ser.write(command)

# =============================================================================
# PHẦN 3: CÁC HÀM XỬ LÝ ẢNH VÀ TÍNH TOÁN
# =============================================================================
def find_lane_lines(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (11, 11), 0)
    edges = cv2.Canny(blur, 95, 180)
    return edges

def birdview_transform(img):
    h, w = img.shape[:2]
    src = np.float32([[0, h], [w, h], [0, h//3], [w, h//3]])
    dst = np.float32([[90, h], [230, h], [-10, 0], [w+10, 0]])
    M = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(img, M, (w, h))

def find_lane_center_error(edge_img):
    h, w = edge_img.shape
    y = int(h * 0.7)
    line = edge_img[y, :]
    center = w // 2
    left = next((x for x in range(center, -1, -1) if line[x] > 0), None)
    right = next((x for x in range(center, w) if line[x] > 0), None)

    if left is None and right is not None:
        left = right - 100
    if right is None and left is not None:
        right = left + 100
    if left is None or right is None:
        return None

    lane_center = (left + right) / 2.0
    error_pixels = lane_center - center
    return error_pixels

def compute_kinematic_velocities(error_pixels, w):
    """ (SỬA ĐỔI) Tính vận tốc dài (m/s) cho mỗi bánh dựa trên mô hình động học. """
    half_fov_rad = math.radians(CAMERA_FOV_DEG / 2)
    angle_per_pixel = half_fov_rad / (w/2)
    delta = error_pixels * angle_per_pixel  # Góc lái (rad)

    if abs(delta) < 1e-3:
        return DESIRED_SPEED, DESIRED_SPEED, delta

    # Bán kính quay của robot (có thể âm hoặc dương)
    # Thêm một giá trị nhỏ (1e-6) để tránh chia cho 0
    R = WHEEL_BASE / math.tan(delta + 1e-6)
    
    # Vận tốc dài cho mỗi bánh
    v_l = DESIRED_SPEED * (1 - (WHEEL_BASE / (2 * R)))
    v_r = DESIRED_SPEED * (1 + (WHEEL_BASE / (2 * R)))
    return v_l, v_r, delta

def map_velocity_to_pwm(v_l, v_r):
    """ (MỚI) Chuyển đổi vận tốc (m/s) sang giá trị PWM (0-255) và giới hạn. """
    # Tính toán giá trị PWM thô dựa trên tỉ lệ
    # Giả định DESIRED_SPEED (m/s) tương ứng với BASE_SPEED_PWM
    pwm_left_raw = (v_l / DESIRED_SPEED) * BASE_SPEED_PWM
    pwm_right_raw = (v_r / DESIRED_SPEED) * BASE_SPEED_PWM
    
    # Giới hạn giá trị PWM trong khoảng [0, MAX_SPEED_PWM]
    # Điều này đảm bảo robot không đi lùi (giá trị >= 0)
    # và không vượt quá tốc độ tối đa
    pwm_left = np.clip(pwm_left_raw, 0, MAX_SPEED_PWM)
    pwm_right = np.clip(pwm_right_raw, 0, MAX_SPEED_PWM)

    return int(pwm_left), int(pwm_right)

# =============================================================================
# PHẦN 4: VÒNG LẶP CHÍNH
# =============================================================================
cap = cv2.VideoCapture(VIDEO_PATH)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Hết video, chạy lại từ đầu.")
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        continue
    
    img = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
    edges = find_lane_lines(img)
    bird = birdview_transform(edges)
    
    # Tính sai số
    err = find_lane_center_error(bird)

    if err is not None:
        # 1. Tính vận tốc lý thuyết (m/s)
        v_l, v_r, steering_angle = compute_kinematic_velocities(err, FRAME_WIDTH)
        
        # 2. (MỚI) Chuyển đổi vận tốc sang PWM
        pwm_l, pwm_r = map_velocity_to_pwm(v_l, v_r)
        
        # 3. (MỚI) Gửi lệnh PWM tới ESP32
        send_motor_command(ser, MOTOR_LEFT_ADDR, FORWARD_DIR, pwm_l)
        send_motor_command(ser, MOTOR_RIGHT_ADDR, FORWARD_DIR, pwm_r)

    else:
        # AN TOÀN: Nếu không thấy làn đường, dừng robot
        v_l, v_r, steering_angle = 0.0, 0.0, 0.0
        pwm_l, pwm_r = 0, 0
        send_motor_command(ser, MOTOR_LEFT_ADDR, FORWARD_DIR, 0)
        send_motor_command(ser, MOTOR_RIGHT_ADDR, FORWARD_DIR, 0)

    # --- Hiển thị thông tin lên ảnh để gỡ lỗi ---
    disp = img.copy()
    cv2.putText(disp, f"Error: {err if err is not None else 'N/A'}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    cv2.putText(disp, f"Angle: {math.degrees(steering_angle):.1f} deg", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    cv2.putText(disp, f"V_L: {v_l:.2f} m/s", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
    cv2.putText(disp, f"V_R: {v_r:.2f} m/s", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
    cv2.putText(disp, f"PWM_L: {pwm_l}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
    cv2.putText(disp, f"PWM_R: {pwm_r}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

    cv2.imshow("Result", disp)
    cv2.imshow("Bird's Eye View", bird)

    if cv2.waitKey(20) & 0xFF == 27: # Nhấn phím ESC để thoát
        break

# =============================================================================
# PHẦN 5: DỌN DẸP TRƯỚC KHI THOÁT
# =============================================================================
print("Đang dừng robot và thoát chương trình...")
# Gửi lệnh dừng cuối cùng để đảm bảo robot không chạy tiếp
send_motor_command(ser, MOTOR_LEFT_ADDR, FORWARD_DIR, 0)
send_motor_command(ser, MOTOR_RIGHT_ADDR, FORWARD_DIR, 0)
time.sleep(0.1) # Đợi một chút để lệnh được gửi đi

cap.release()
if ser and ser.is_open:
    ser.close()
    print("Đã đóng cổng Serial.")
cv2.destroyAllWindows()