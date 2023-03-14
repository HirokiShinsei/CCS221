#Group - 6
#Lord Patrick Raizen Togonon
#Matthew Andrei Valencia
#Cyril Reynold Trojillo

import numpy as np
import cv2 
import matplotlib.pyplot as plt
import streamlit as st

#The following functions are for each spefic transformations to be applied to the image.

# def translation(img, tx, ty):
#     rows, cols = img.shape[:2]
#     m_translation_ =np.float32([[1, 0, tx],
#                                 [0, 1, ty],
#                                 [0, 0, 1]])

#     translated_img_= cv2.warpPerspective(img, m_translation_,(cols, rows))
#     return translated_img_

def translation(img, tx, ty):
    rows, cols = img.shape[:2]
    m_translation_ =np.float32([[1, 0, tx],
                                [0, 1, ty]])

    translated_img_= cv2.warpAffine(img, m_translation_,(cols, rows))
    return translated_img_

#Main

st.title("Test Run!")
oldX = [2,3,4,5,6]
oldY = [5,6,7,8,9]
BXnew = []
BYnew = []
tx_1 = st.number_input("Enter X Value: ")
ty_1 = st.number_input("Enter Y Value: ")

incr = 50

# Initialize variables
oldX = [10, 20, 30, 40, 50]
oldY = [15, 25, 35, 45, 55]
tx_1 = 5
ty_1 = 10
incr = 0
BXnew = []
BYnew = []

st.title("Image Transformation")

#Image Upload
img_file = st.file_uploader("Choose a file")

if img_file is not None:
    
    image_bytes = np.asarray(bytearray(img_file.read()), dtype=np.uint8)
    img = cv2.imdecode(image_bytes, 1)
    st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption="Uploaded Image", use_column_width=True)
    st.write("Image shape:", img.shape)
    
    # Create Streamlit app
    st.write("Calculate new values of BXnew and BYnew")
    for i in range(5):
        incr += 0
        BXnew_n = oldX[i] + tx_1 + incr
        BYnew_n = oldY[i] + ty_1 + incr
        BXnew.append(BXnew_n)
        BYnew.append(BYnew_n)
        st.write(f"i = {i}: BXnew_n = {BXnew_n}, BYnew_n = {BYnew_n}")

    # Display results
    st.write("Final results:")
    st.write(f"BXnew = {BXnew}")
    st.write(f"BYnew = {BYnew}")
    

    #Old Coordinates
    old_translated_img_1 = translation(img, oldX[0], oldY[0])
    old_translated_img_2 = translation(img, oldX[1], oldY[1])
    old_translated_img_3 = translation(img, oldX[2], oldY[2])
    old_translated_img_4 = translation(img, oldX[3], oldY[3])
    old_translated_img_5 = translation(img, oldX[4], oldY[4])

    #New Coordinates
    new_translated_img_1 = translation(img, BXnew[0], BYnew[0])
    new_translated_img_2 = translation(img, BXnew[1], BYnew[1])
    new_translated_img_3 = translation(img, BXnew[2], BYnew[2])
    new_translated_img_4 = translation(img, BXnew[3], BYnew[3])
    new_translated_img_5 = translation(img, BXnew[4], BYnew[4])

    #The following functions plots where the specific transformation be placed on a 2 column by 4 rows figure.
    fig, axs = plt.subplots(2, 5, figsize=(18, 14))
    fig = plt.gcf()
    fig.canvas.manager.set_window_title('Quiz') #This line sets the window title to OpenCV Transformations

    axs[0, 0].imshow(old_translated_img_1)
    axs[0, 0].set_title("Number #1")
    axs[0, 0].axis("off")

    axs[0, 1].imshow(old_translated_img_2)
    axs[0, 1].set_title("Number #2")
    axs[0, 1].axis("off")

    axs[0, 2].imshow(old_translated_img_3)
    axs[0, 2].set_title("Number #3")
    axs[0, 2].axis("off")

    axs[0, 3].imshow(old_translated_img_4)
    axs[0, 3].set_title("Number #4")
    axs[0, 3].axis("off")

    axs[0, 4].imshow(old_translated_img_5)
    axs[0, 4].set_title("Number #5")
    axs[0, 4].axis("off")

    axs[1, 0].imshow(new_translated_img_1)
    axs[1, 0].set_title("New")
    axs[1, 0].axis("off")

    axs[1, 1].imshow(new_translated_img_2)
    axs[1, 1].set_title("New")
    axs[1, 1].axis("off")

    axs[1, 2].imshow(new_translated_img_3)
    axs[1, 2].set_title("New")
    axs[1, 2].axis("off")

    axs[1, 3].imshow(new_translated_img_4)
    axs[1, 3].set_title("New")
    axs[1, 3].axis("off")

    axs[1, 4].imshow(new_translated_img_5)
    axs[1, 4].set_title("New")
    axs[1, 4].axis("off")

    st.pyplot(fig)