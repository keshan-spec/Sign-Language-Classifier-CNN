from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import os


# generates transformed images for more data
def generate_images(img, prefix='sample'):
    if not os.path.exists("generated_images"):
        os.mkdir("generated_images")
    os.chdir("generated_images")
    
    # creates a data generator object that transforms images
    datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')
    try:
        # pick an image to transform
        img = image.img_to_array(img)  # convert image to numpy array
        img = img.reshape((1,) + img.shape)  # reshape image
        # this loops runs forever until we break, saving images to current directory with specified prefix
        i = 0
        # this loops runs forever until we break, saving images to current directory with specified prefix
        for batch in datagen.flow(img, save_prefix='test', save_format='jpeg'):
            # plt.figure(i)
            # plot = plt.imshow(image.img_to_array(batch[0]))
            cv.imwrite(f'{i}.jpg', image.img_to_array(batch[0]))
            i += 1
            if i > 4:  # show 4 images
                break
        plt.show()
    except Exception as e:
        print(f"[ERROR]  {e}")


if __name__ == "__main__":
    import tensorflow as tf
    img_data = np.random.random(size=(100, 100, 3))
    img = tf.keras.preprocessing.image.array_to_img(img_data)
    array = tf.keras.preprocessing.image.img_to_array(img)

    generate_images(img)
