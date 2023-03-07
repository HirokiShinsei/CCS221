import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.write('Choose where to plot in a 3x3 grid')
st.write(' ')
st.write('1 2 3')
st.write('4 5 6')
st.write('7 8 9')

choice = st.selectbox('Choice:', [1, 2, 3, 4, 5, 6, 7, 8, 9])

if choice == 1:
    two_d_arr = np.zeros((3, 3)) # create a 10x10 array of zeros

    # modify the values in the array as needed


plt.imshow(two_d_arr, interpolation='none', cmap='winter')
st.pyplot()
