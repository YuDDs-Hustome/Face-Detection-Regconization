import dlib
import cv2
import numpy as np
import os
import necessaryMothod as  nm

# Tải mô hình phát hiện khuôn mặt và mô hình trích xuất đặc trưng khuôn mặt
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('./cores/shape_predictor_68_face_landmarks.dat')
face_rec_model = dlib.face_recognition_model_v1('./cores/dlib_face_recognition_resnet_model_v1.dat')

# Hàm để trích xuất đặc trưng khuôn mặt từ một ảnh
def extract_face_features(image_path):
    try:
        image = cv2.imread(image_path)
        cv2.imshow('Landmarks', image)
        cv2.waitKey(1000)
        cv2.destroyAllWindows()
    except:
        print("Exception!")

    # img = [image]
    # nm.display(img)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detector(gray, 1)
    if len(faces) == 0:
        print("None!")
        return None
    shape = predictor(gray, faces[0])
    face_descriptor = face_rec_model.compute_face_descriptor(image, shape)
    return np.array(face_descriptor)

#######################################################################################################
##### Learning part
## Lưu các đặc trưng khuôn mặt từ một loạt ảnh của cùng một người
# face_features_list = []
# image_paths = f"./resource/video/gaeul_face/"  # Đường dẫn đến các ảnh của cùng một người
# listImg = []
# rr = 0
# features = None
# for image_file in os.listdir(image_paths):
#     # rr += 1
#     # if rr > 20:
#     #     break
#     if image_file.endswith('.jpg'):
#         features = extract_face_features(image_paths + image_file)
#     if features is not None:
#         face_features_list.append(features)

# ## Chuyển đổi danh sách đặc trưng thành mảng numpy và lưu vào file
# face_features_array = np.array(face_features_list)
# print(len(face_features_array))
# print(face_features_array.shape)
# np.save('face_features.npy', face_features_array)

#######################################################################################################
##### Testing part
## Tải các đặc trưng khuôn mặt đã lưu
saved_face_features = np.load('face_features.npy')

## Hàm để so sánh khuôn mặt mới với các đặc trưng đã lưu
def is_same_person(new_image_path, saved_features):
    new_face_features = extract_face_features(new_image_path)
    if new_face_features is None:
        return False
    distances = np.linalg.norm(saved_features - new_face_features, axis=1)
    print(distances)
    print("Max: ", max(distances))
    print("Min: ", min(distances))
    avg_distance = np.mean(distances)
    print(avg_distance)
    threshold = 0.6  # Ngưỡng để xác định mức độ giống nhau
    return avg_distance < threshold 

## Kiểm tra xem ảnh mới có phải của người đó hay không
new_image_path = './resource/imgs/ive/ive_4.jpg'
if is_same_person(new_image_path, saved_face_features):
    print("Ảnh mới là của cùng một người.")
else:
    print("Ảnh mới không phải của cùng một người.")
