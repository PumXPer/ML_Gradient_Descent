import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.title("ML - Gradient Descent")

# Create fakedata
N = 100
D = 100

x = np.random.rand(N) * D
y = 2 * x

# noise data
na = 50
fakenoise = (np.random.rand(len(y)) - 0.5) * na
y = y + fakenoise

# plot data 1
st.header('Fake data')
st.markdown('data from random 200 data X and Y 100 each and calculate')
st.markdown('ข้อมูลที่มาจากการสุ่มจำนวน 200 ข้อมูล X และ Y อย่างละ 100 แล้วนำมาคำนวณ')
fig, ax = plt.subplots()
ax.scatter(x, y)

st.pyplot(fig)

# linear regression
w = 0
h = w * x


def Loss_func(w):
    h = w * x
    mse = np.mean((h - y) ** 2)
    return mse


# create empty list to record error for each W
mseList = []

# random wList
wList = np.arange(-200, 240) / 10

# loop for every w in wList
for w in wList:
    mse = Loss_func(w)
    mseList.append(mse)

# plot data 2
st.header('Loss function (การหาค่าความชัน)')
st.markdown('ค่าความชันของกราฟ คือ ค่าความคลาดเคลื่อนระหว่างค่าจริงกับค่าที่คำนวณได้')
st.markdown('เมื่อเรามีค่าความชัน จะสามารถหาค่า W ที่เหมาะสมได้')
st.latex(r''' \frac{1}{n} \sum_{i=1}^{n} (h_{i} - y_{i})^{2} ''')
fig, ax = plt.subplots()
ax.scatter(wList, mseList)

st.pyplot(fig)

st.header('Training Progress')
st.markdown('การเทรนโมเดล คือ การหาค่า W ที่เหมาะสมที่สุด โดยใช้ Gradient Descent')
st.markdown('ค่าเริ่มต้น w = -10000, learning_rate หรือ alpha= 0.0001')
st.markdown(':red[แนะนำ] จำนวนรอบการเทรน 50 รอบ')
number = st.number_input('Insert a round train', value=0, step=1, format='%d')
number = int(number)
st.write('The current round is', number)
# Gradient Descent


def train_model(iter=0, w=-10000, alpha=0.0001):
    wList = [w]
    mseList = [np.mean((y - w * x) ** 2)]

    fig, ax1 = plt.subplots()

    ax1.scatter(x, y)
    line1, = ax1.plot(x, x * 0, c='r')
    ax1.set_title('Gradient Descent')

    placeholder1 = st.empty()

    for i in range(iter):
        w = w - alpha * np.mean((w * x - y) * x)

        wList.append(w)
        mseList.append(np.mean((y - w * x) ** 2))

        line1.set_ydata(x * w)

        placeholder1.pyplot(fig)
        plt.pause(0.1)

    return wList, mseList

w_list, loss_list = train_model(iter=number)
st.write('ค่า W ที่เหมาะสมที่สุดคือ', int(w_list[-1]))