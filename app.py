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
st.header('Gradient descent')
st.markdown('Gradient Descent เป็นอัลกอริทึมที่ใช้ในการหาค่า weight ที่เหมาะสมให้กับโมเดลของ Machine Learning เพื่อให้มีค่า error หรือค่าความคลาดเคลื่อนลดลงได้มากที่สุด อัลกอริทึมนี้มักถูกนำมาใช้ใน Linear Regression, Logistic Regression, และ Neural Networks ซึ่งเป็นแบบจำลองที่แพร่หลายใน Machine Learning')
st.markdown('เพื่อให้เข้าใจการทำงานของ Gradient Descent ได้ง่ายขึ้น ควรทำความเข้าใจกับ Linear Regression ก่อน ซึ่ง Linear Regression เป็นกระบวนการทำนายโดยสร้างเส้นตรงบนจุดข้อมูล โดยเส้นตรงนี้สร้างจากสมการเส้นตรง y = mx + c โดยที่ m และ c เป็นค่า weight ที่ Gradient Descent จะค้นหา')
st.latex(r''' y = mx + c ''')

st.header('Loss function (การหาค่าความชัน)')
st.markdown('Gradient Descent ทำงานโดยการทำ optimization บนค่าความชันและค่าคงที่ของสมการเส้นตรงใน Linear Regression แบบง่ายๆคือการปรับค่า weight ใหม่ให้เหมาะสมให้มากที่สุดเพื่อให้มีค่า error หรือค่าความคลาดเคลื่อนลดลงได้มากที่สุด  สูตร Gradient Descent สำหรับคำนวณค่า weight ใหม่มีดังนี้')
st.latex(r''' w = w - \alpha \frac{d}{dw} \frac{1}{n} \sum_{i=1}^{n} (h_{i} - y_{i})^{2} ''')
# st.markdown('เมื่อ w คือค่า weight ที่เราต้องการหา ซึ่งเป็นค่าที่เราจะปรับให้เหมาะสม โดยการลดค่าความคลาดเคลื่อนของโมเดล ซึ่งเราสามารถหาค่าความคลาดเคลื่อน(MSE)ได้จากสมการดังนี้')
# st.latex(r''' \frac{1}{n} \sum_{i=1}^{n} (h_{i} - y_{i})^{2} ''')


fig, ax = plt.subplots()
ax.scatter(wList, mseList)

st.pyplot(fig)

st.header('Training Progress')
st.markdown('ในการทำงานของ Gradient Descent รอบแรก โดยปกติจะเริ่มด้วยการสุ่มค่า weight และวัดประสิทธิภาพของโมเดลที่ได้จากค่า weight ด้วยค่าความคลาดเคลื่อนจากฟังก์ชันความคลาดเคลื่อน จากนั้น Gradient Descent จะปรับค่า weight ใหม่โดยลดลง เราจะทำซ้ำกระบวนการดังกล่าวไปเรื่อยๆ โดยการปรับค่า weight ใหม่ในแต่ละรอบ จนกว่าเราจะพบจุดต่ำสุด ซึ่งอาจจะเป็นคำตอบที่เหมาะสมที่สุดสำหรับโมเดลของเรา')
st.markdown('ค่าเริ่มต้น w = -10000, learning_rate หรือ alpha= 0.0001')
st.markdown(':red[แนะนำ] จำนวนรอบการเทรน 50 รอบ')
number = st.number_input('Insert a round train', value=0, step=1, format='%d')
number = int(number)
st.write('The current round is', str(number))
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
st.write('ค่า W ที่เหมาะสมที่สุดคือ', str(w_list[-1]))