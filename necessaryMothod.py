import numpy as np
import matplotlib.pyplot as plt

def display(img):
    if len(img) > 8:
        print("Number of image lower than 8!")
        return
    else:
        fig = plt.figure(figsize=(10, 15))
        rows = len(img)//2 + len(img)%2
        columns = 2

        for i in range(len(img)):
            fig.add_subplot(rows, columns, i+1)
            plt.axis('off')
            plt.imshow(img[i])

        plt.show()
