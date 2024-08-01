import cv2
import dlib
import numpy as np
import os

#######################################################################################################3

detector = dlib.get_frontal_face_detector()
predictor_path = './cores/shape_predictor_68_face_landmarks.dat'
face_rec_model = dlib.face_recognition_model_v1('./cores/dlib_face_recognition_resnet_model_v1.dat')
predictor = dlib.shape_predictor(predictor_path)

offset_margin = 10 # Cutting bigger area around face

#######################################################################################################

target_size = (320, 320)
def resizeImg(img, I):
    h, w = img.shape[:2]
    aspect_ratio = w / h
    new_w, new_h = target_size
    if aspect_ratio > 1:
        # Hình ảnh rộng hơn chiều cao, thay đổi kích thước theo chiều rộng
        new_w = target_size[0]
        new_h = int(new_w / aspect_ratio)
    else:
        # Hình ảnh cao hơn chiều rộng, thay đổi kích thước theo chiều cao
        new_h = target_size[1]
        new_w = int(new_h * aspect_ratio)
    resized_img = cv2.resize(img, (new_w, new_h))
    
    # Black background image
    final_img = np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8)

    # Tính toán vị trí để đặt ảnh đã thay đổi kích thước vào ảnh nền đen
    x_offset = (target_size[0] - new_w) // 2
    y_offset = (target_size[1] - new_h) // 2

    # Đặt ảnh đã thay đổi kích thước vào ảnh nền trắng
    final_img[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized_img
    # final_img = cv2.cvtColor(final_img, cv2.COLOR_BGR2GRAY)

    cv2.imwrite(f'./resource/video/{name}_face/{name}{I}.jpg', final_img)

#######################################################################################################

name = "gaeul"
for ite in range(3):
    video_source = f"./resource/video/{name}_{ite}.mp4"
    video_capture = cv2.VideoCapture(video_source)
    i = len(os.listdir(f"./resource/video/{name}_face/"))
    time = 0
    step = 10
    while True:
        # Đọc khung hình từ video
        ret, frame = video_capture.read()
        if not ret:
            print("Finish! [Wrong Path!]}")
            break
        try:
            img = frame.copy()
        except:
            print("Exception!")

        # Chuyển đổi khung hình sang màu xám
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Phát hiện khuôn mặt
        faces = detector(gray, 1)
        # print("isDetected?: ", len(faces))

        # Xử lý từng khuôn mặt phát hiện được
        all_y = []
        all_x = []    
        for face in faces:
            # Vẽ hình chữ nhật quanh khuôn mặt
            x, y, w, h = (face.left(), face.top(), face.right(), face.bottom())
            cv2.rectangle(img, (x, y), (w, h), (0, 255, 0), 1)  
            # Lấy điểm đặc trưng cho khuôn mặt
            landmarks = predictor(gray, face)
    
            for n in range(68):
                x, y = landmarks.part(n).x, landmarks.part(n).y
                all_x.append(landmarks.part(n).x)
                all_y.append(landmarks.part(n).y)
                cv2.circle(img, (x, y), 1, (0, 0, 255), -1)

        if (len(faces)!=0) and (time%step==0):
            yc = int(min(y, min(all_y))) - offset_margin
            hc = int(max(h, max(all_y))) + offset_margin
            xc = int(min(x, min(all_x))) - offset_margin
            wc = int(max(w, max(all_x))) + offset_margin
            cv2.rectangle(img, (xc, yc), (wc, hc), (255, 0, 0), 1)
            cropped_image = frame[yc:hc, xc:wc]
            resizeImg(cropped_image, i)
            i += 1
            print("-----------Saved!-----------")

        # Hiển thị khung hình với các khuôn mặt phát hiện được
        cv2.imshow('Video Face Detection', img)
        # Thoát khi nhấn phím 'q'
        time += 1
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
