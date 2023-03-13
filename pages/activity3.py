import streamlit as st
import numpy as np
import cv2
from matplotlib import pyplot as plt
from PIL import Image

#Functions for image transformations (translation, rotation, shear, reflect)
def translation(img, tx, ty):
    rows, cols = img.shape[:2]
    m_translation_ = np.float32([[1, 0, tx],
                                 [0, 1, ty],
                                 [0, 0, 1]])

    translated_img_ = cv2.warpPerspective(img, m_translation_, (cols, rows))
    return translated_img_


def rotation(img, rotx):
    angle = np.radians(10)
    rows, cols = img.shape[:2]
    m_rotation_ = np.float32([[np.cos(angle), -(np.sin(angle)), 0],
                              [np.sin(angle), np.cos(angle), 0],
                              [0, 0, 1]])

    m_rotation_ = cv2.getRotationMatrix2D((cols/2, rows/2), rotx, 1)
    rotated_img_ = cv2.warpAffine(img, m_rotation_, (cols, rows))
    return rotated_img_


def scaling_img(img, scaleX, scaleY):
    rows, cols = img.shape[:2]
    m_scaling_ = np.float32([[scaleX, 0, 0],
                            [0, scaleY, 0],
                             [0, 0, 1]])
    scaled_img_ = cv2.warpPerspective(img, m_scaling_, (cols*2, rows*2))
    return scaled_img_


def reflection_v(img):
    rows, cols = img.shape[:2]
    m_reflection_ = np.float32([[1, 0, 0],
                                [0, -1, rows],
                                [0, 0, 1]])
    reflected_img_ = cv2.warpPerspective(
        img, m_reflection_, (int(cols), int(rows)))
    return reflected_img_


def reflection_h(img):
    rows, cols = img.shape[:2]
    m_reflection_ = np.float32([[-1, 0, cols],
                                [0, 1, 0],
                                [0, 0, 1]])
    reflected_img_ = cv2.warpPerspective(
        img, m_reflection_, (int(cols), int(rows)))
    return reflected_img_


def shear_X(img, shearX):
    rows, cols = img.shape[:2]
    m_shearing_x = np.float32([[1, shearX, 0],
                               [0, 1, 0],
                               [0, 0, 1]])

    sheared_img_x = cv2.warpPerspective(
        img, m_shearing_x, (int(cols*1.5), int(rows*1.5)))
    return sheared_img_x


def shear_Y(img, shearY):
    rows, cols = img.shape[:2]
    m_shearing_x = np.float32([ [1, 0, 0],
                                [shearY, 1, 0],
                                [0, 0, 1]])

    sheared_img_x = cv2.warpPerspective(
        img, m_shearing_x, (int(cols*1.5), int(rows*1.5)))
    return sheared_img_x


def main():
    
    #Image Upload
    img_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])
    
    
    #Main Process
    if img_file is not None:
        img = Image.open(img_file)
        st.image(img, caption="Uploaded Image", use_column_width=True)
        st.write("Image shape:", img.shape)
        
        # The following lines takes user input for shifting X, Y, and rotation directions, shear value(skew) whether it be to the right or left.
        tx = st.slider("Enter the value to shift in X-axis", -100, 100, 0)
        ty = st.slider("Enter the value to shift in Y-axis", -100, 100, 0)
        rotx = st.slider("Enter value to rotate the image in degrees", -180, 180, 0)
        scaleX = st.slider("Enter the value to scale in X-Axis", 0.1, 5.0, 1.0)
        scaleY = st.slider("Enter the value to scale in Y-Axis", 0.1, 5.0, 1.0)
        shearX = st.slider("Enter the value to shear in X-Axis", -1.0, 1.0, 0.0, step=0.1)
        shearY = st.slider("Enter the value to shear in Y-Axis", -1.0, 1.0, 0.0, step=0.1)

        # The following lines calls each of the functions for specific transformations of the image.
        translated_img_ = translation(img, tx, ty)
        rotated_img_ = rotation(img, rotx)
        scaled_img_ = scaling_img(img, scaleX, scaleY)
        reflected_h = reflection_h(img)
        reflected_v = reflection_v(img)
        sheared_img_x = shear_X(img, shearX)
        sheared_img_y = shear_Y(img, shearY)

        # The following functions plots where the specific transformation be placed on a 2 column by 4 rows figure.
        fig, axs = plt.subplots(2, 4, figsize=(18, 14))
        fig = plt.gcf()
        # This line sets the window title to OpenCV Transformations
        fig.canvas.manager.set_window_title('OpenCV Transformations')

        axs[0, 0].imshow(img)
        axs[0, 0].set_title("Original Image")
        axs[0, 0].axis("off")

        axs[0, 1].imshow(translated_img_)
        axs[0, 1].set_title("Translated Image")
        axs[0, 1].axis("off")

        axs[0, 2].imshow(rotated_img_)
        axs[0, 2].set_title("Rotated Image")
        axs[0, 2].axis("off")

        axs[0, 3].imshow(scaled_img_)
        axs[0, 3].set_title("Scaled Image")
        axs[0, 3].axis("off")

        axs[1, 0].imshow(reflected_h)
        axs[1, 0].set_title("Reflected Image (Horizontal)")
        axs[1, 0].axis("off")

        axs[1, 1].imshow(reflected_v)
        axs[1, 1].set_title("Reflected Image (Vertical)")
        axs[1, 1].axis("off")

        axs[1, 2].imshow(sheared_img_x)
        axs[1, 2].set_title("Sheared Horizontally(X)")
        axs[1, 2].axis("off")

        axs[1, 3].imshow(sheared_img_y)
        axs[1, 3].set_title("Sheared Vertically(Y)")
        axs[1, 3].axis("off")
        
        st.pyplot(fig)
    
if __name__ == "main":
    main()


