import pickle
import numpy as np
# Đọc các điểm đặc trưng từ tệp
with open('landmarks.dat', 'rb') as file:
    loaded_landmarks = pickle.load(file)


loaded_landmarks = np.array(loaded_landmarks)
print(loaded_landmarks)
print(type(loaded_landmarks))
print(loaded_landmarks.shape)
loaded_landmarks = loaded_landmarks.reshape([68, 2])
print(loaded_landmarks.shape)