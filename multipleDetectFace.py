import cv2
import dlib

# Khong the thuc hien duoc

# Đọc ảnh
image_path = './resource/imgs/ive/ive_8.jpg'
image = cv2.imread(image_path)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Tải mô hình phát hiện khuôn mặt và mô hình điểm đặc trưng
detector = dlib.get_frontal_face_detector()
predictor_path = './cores/shape_predictor_68_face_landmarks.dat'
predictor = dlib.shape_predictor(predictor_path)

# Phát hiện tất cả các khuôn mặt trong ảnh
faces = detector(gray, 1)
print("Kich thuoc tep nhan dien: ", len(faces))

# Xử lý từng khuôn mặt phát hiện được
for face in faces:
    # Lấy điểm đặc trưng cho khuôn mặt hiện tại
    landmarks = predictor(gray, face)
    
    # Vẽ hình chữ nhật quanh khuôn mặt
    x, y, w, h = (face.left(), face.top(), face.right(), face.bottom())
    cv2.rectangle(image, (x, y), (w, h), (0, 255, 0), 1)
    
    # Vẽ các điểm đặc trưng trên khuôn mặt
    for n in range(68):
        x, y = landmarks.part(n).x, landmarks.part(n).y
        cv2.circle(image, (x, y), 2, (0, 0, 255), -1)

# Hiển thị ảnh với các khuôn mặt và điểm đặc trưng
cv2.imshow('Face Landmarks', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
