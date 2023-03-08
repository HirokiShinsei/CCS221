# Work by Group 6
# Members:
#   Lord Patrick Raizen Togonon
#   Cyril Reynold Trojillo
#   Matthew Andrei Valencia


import streamlit as st
import matplotlib.pyplot as plt

#DDA Line algorithm with midpoint
def DDALineMid(x1, y1, x2, y2, color):
    
    dx = x2 - x1
    dy = y2 - y1

    #calculate steps required for generating pixels

    steps = abs(dx) if abs(dx) > abs (dy) else abs(dy)

    #calculate increment in x & y for each steps

    Xinc = float(dx / steps)
    Yinc = float(dy / steps)

    midX = ((x1 + x2)/2)
    midY = ((y1 + y2)/2)

    for i in range(0, int(steps +1 )):

        #draw pixels

        plt.plot(int(x1), int (y1), color)
        x1 += Xinc
        y1 += Yinc
    
    plt.scatter(midX, midY, color = 'red') 


#DDA Line without the midpoint
def DDALine(x1, y1, x2, y2, color):
    
    dx = x2 - x1
    dy = y2 - y1

    #calculate steps required for generating pixels

    steps = abs(dx) if abs(dx) > abs (dy) else abs(dy)

    #calculate increment in x & y for each steps

    Xinc = float(dx / steps)
    Yinc = float(dy / steps)

    for i in range(0, int(steps +1 )):

        #draw pixels

        plt.plot(int(x1), int (y1), color)
        x1 += Xinc
        y1 += Yinc
    

#bresenham's line algorithm with midpoint
def bresenhamLineMid(x1, y1, x2, y2, color):
    dx = x2 - x1
    dy = y2 - y1

    # determine the sign of the change in x and y
    sx = 1 if dx > 0 else -1
    sy = 1 if dy > 0 else -1

    # calculate the absolute value of the change in x and y
    dx = abs(dx)
    dy = abs(dy)

    # initialize the decision parameter
    p = 2 * dy - dx

    for x, y in zip(range(x1, x2+sx, sx), range(y1, y2+sy, sy)):
        # plot the pixel
        plt.plot(x, y, color)
        
        # update the decision parameter
        if p >= 0:
            y = y + sy
            p = p + 2 * (dy - dx)
        else:
            p = p + 2 * dy

    midX = (x1 + x2) // 2
    midY = (y1 + y2) // 2
     
    plt.scatter(midY, midX)


#bresenham's line algorithm without the midpoint
def bresenhamLine(x1, y1, x2, y2, color):
    dx = x2 - x1
    dy = y2 - y1

    # determine the sign of the change in x and y
    sx = 1 if dx > 0 else -1
    sy = 1 if dy > 0 else -1

    # calculate the absolute value of the change in x and y
    dx = abs(dx)
    dy = abs(dy)

    # initialize the decision parameter
    p = 2 * dy - dx

    for x, y in zip(range(x1, x2+sx, sx), range(y1, y2+sy, sy)):
        # plot the pixel
        plt.plot(x, y, color)
        
        # update the decision parameter
        if p >= 0:
            y = y + sy
            p = p + 2 * (dy - dx)
        else:
            p = p + 2 * dy

def main():
    x = st.number_input('X1', min_value=1, max_value=100, value=15, step=1)
    y = st.number_input('Y1', min_value=1, max_value=100, value=15, step=1)

    xEnd = st.number_input('X2', min_value=1, max_value=100, value=20, step=1)
    yEnd = st.number_input('Y2', min_value=1, max_value=100, value=20, step=1)

    color = "b."

    plt.subplot(2,2,1)
    DDALineMid(x, y, xEnd, yEnd, color)
    plt.title("DDA Line w/ midpoint")

    plt.subplot(2,2,2)
    DDALine(x, y, xEnd, yEnd, color)
    plt.title("DDA Line")

    color = "y."

    plt.subplot(2,2,3)
    bresenhamLineMid(x, y, xEnd, yEnd, color)
    plt.title("Bresenham's Line w/ midpoint")

    plt.subplot(2,2,4)
    bresenhamLine(x, y, xEnd, yEnd, color)
    plt.title("Bresenham's Line")

    st.pyplot(plt)

if __name__ == '__main__':
    main()
