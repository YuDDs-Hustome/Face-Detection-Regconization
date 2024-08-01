# Tải các đặc trưng khuôn mặt đã lưu
saved_face_features = np.load('face_features.npy')

# Hàm để so sánh khuôn mặt mới với các đặc trưng đã lưu
def is_same_person(new_image_path, saved_features):
    new_face_features = extract_face_features(new_image_path)
    if new_face_features is None:
        return False
    distances = np.linalg.norm(saved_features - new_face_features, axis=1)
    avg_distance = np.mean(distances)
    threshold = 0.6  # Ngưỡng để xác định mức độ giống nhau
    return avg_distance < threshold

# Kiểm tra xem ảnh mới có phải của người đó hay không
new_image_path = 'new_image.jpg'
if is_same_person(new_image_path, saved_face_features):
    print("Ảnh mới là của cùng một người.")
else:
    print("Ảnh mới không phải của cùng một người.")
