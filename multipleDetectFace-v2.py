import face_recognition
import cv2
import matplotlib.pyplot as plt

# Đọc ảnh
image_path = './resource/imgs/ive/ive_1.jpg'
image = face_recognition.load_image_file(image_path)

# Tìm tất cả các khuôn mặt trong ảnh
face_locations = face_recognition.face_locations(image)

# Hiển thị ảnh với các khuôn mặt được nhận diện
# for (top, right, bottom, left) in face_locations:
#     cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 1)


(top, right, bottom, left) = face_locations[1]
cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 1)

# Hiển thị ảnh
plt.imshow(image)
plt.show()
