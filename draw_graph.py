#%% 
import matplotlib.pyplot as plt
import numpy as np

x_axis = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
y_axis = [0.5268, 0.5718, 0.5718, 0.5549, 0.5718, 0.5577, 0.5324, 0.5662, 0.5577, 0.5944 ]

x_axis = [
    10, 
    20, 
    30, 
    40, 
    50, 
    60, 
    70, 
    80, 
    90, 
    100
]
y_axis_1 = np.array([
    0.5268, 
    0.5718, 
    0.5718, 
    0.5549, 
    0.5718, 
    0.5577, 
    0.5324, 
    0.5662, 
    0.5577, 
    0.5944,
])

y_axis_2 = np.array([
    0.5831,
    0.538,
    0.5662,
    0.5746,
    0.5803,
    0.5831,
    0.6141,
    0.6169,
    0.6197,
    0.6113,
])

y_axis = np.average([y_axis_1, y_axis_2], axis=0)


plt.plot(x_axis, y_axis)
plt.title('Personality Trait Detection Performance')
plt.xlabel('Training data (percent)')
plt.ylabel('Avg. Accuracy (5 traits)')
plt.show()

# %%

import matplotlib.pyplot as plt

x_axis = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
emotion = [0.4716, 0.5069, 0.5376, 0.5469, 0.5484, 0.553, 0.553,  0.5545, 0.5745, 0.5668]
sentiment = [0.5376, 0.5822, 0.5945, 0.6129, 0.6206, 0.6237, 0.6221, 0.6206, 0.6482, 0.6498]

plt.plot(x_axis, emotion, label='Emotion')
plt.plot(x_axis, sentiment, '-.', label='Sentiment')

plt.title('Emotion & Sentiment Detection Performance')
plt.xlabel('Training data (percent)')
plt.legend()
plt.ylabel('Accuracy')
plt.show()
# %%
