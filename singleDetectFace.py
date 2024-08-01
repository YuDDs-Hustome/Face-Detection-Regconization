import cv2
import dlib
import pickle

# Đọc ảnh
image = cv2.imread('./resource/video/gaeul_face/gaeul1.jpg')

# Chuyển đổi ảnh sang grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Tạo bộ phát hiện khuôn mặt và bộ dự đoán điểm đặc trưng
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('./cores/shape_predictor_68_face_landmarks.dat')

# Phát hiện các khuôn mặt trong ảnh
faces = detector(gray)

# Tạo danh sách để lưu các điểm đặc trưng
landmarks_list = []

# Duyệt qua từng khuôn mặt và dự đoán các điểm đặc trưng
for face in faces:
    landmarks = predictor(gray, face)
    face_landmarks = [(landmarks.part(n).x, landmarks.part(n).y) for n in range(68)]
    landmarks_list.append(face_landmarks)

# Lưu các điểm đặc trưng vào tệp
with open('landmarks.dat', 'wb') as file:
    pickle.dump(landmarks_list, file)

# Hiển thị ảnh với các điểm đặc trưng
for landmarks in landmarks_list:
    for (x, y) in landmarks:
        cv2.circle(image, (x, y), 2, (255, 0, 0), -1)

cv2.imshow('Landmarks', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
