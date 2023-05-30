import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.header("Gradient Descent")

# Create fakedata
N = 100
D = 100

x = np.random.rand(N) * D
y = 2 * x

# noise data
na = 50
fakenoise = (np.random.rand(len(y)) - 0.5) * na
y = y + fakenoise

# plot data
fig, (ax1, ax2) = plt.subplots(1, 2)

ax1.scatter(x, y)
line1, = ax1.plot(x, x * 0, c='r')
ax1.set_title('Linear Regression')

mseList = []

# random wList
wList = np.arange(-200, 240) / 10

for w in wList:
    mse = np.mean((y - w * x) ** 2)
    mseList.append(mse)

ax2.scatter(wList, mseList)
line2, = ax2.plot(wList, mseList)
ax2.set_xlabel('Weight (w)')
ax2.set_ylabel('Mean Squared Error')
ax2.set_title('Training Progress')


# Gradient Descent
def train_model(iter=50, w=-10000, alpha=0.0001):
    wList = [w]
    mseList = [np.mean((y - w * x) ** 2)]

    placeholder1 = st.empty()

    for i in range(iter):
        w = w - alpha * np.mean((w * x - y) * x)

        wList.append(w)
        mseList.append(np.mean((y - w * x) ** 2))

        line1.set_ydata(x * w)
        line2.set_data(wList, mseList)

        placeholder1.pyplot(fig)
        plt.pause(0.1)

    return wList, mseList


w_list, loss_list = train_model()
