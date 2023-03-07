import streamlit as st
import numpy as np 
import matplotlib.pyplot as plt 

st.title('Plot on a 3x3 Grid')
st.write('Choose where to plot in a 3x3 grid')

# Define the layout of the grid
grid = [[1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]]

# Display the grid
for row in grid:
    st.write(row)

# Get user's choice
choice = st.number_input('Enter a number from 1 to 9', min_value=1, max_value=9, step=1)

# Create an empty 2D array
two_d_arr = np.zeros((3, 3))

# Update the corresponding element based on user's choice
if choice == 1:
    two_d_arr[0][0] = 1
elif choice == 2:
    two_d_arr[0][1] = 1
elif choice == 3:
    two_d_arr[0][2] = 1
elif choice == 4:
    two_d_arr[1][0] = 1
elif choice == 5:
    two_d_arr[1][1] = 1
elif choice == 6:
    two_d_arr[1][2] = 1
elif choice == 7:
    two_d_arr[2][0] = 1
elif choice == 8:
    two_d_arr[2][1] = 1
elif choice == 9:
    two_d_arr[2][2] = 1

# Display the 2D array
st.write('The 2D array:', two_d_arr)

# Plot the 2D array
plt.imshow(two_d_arr, interpolation='none', cmap='winter')
plt.axis('off')
st.pyplot()
