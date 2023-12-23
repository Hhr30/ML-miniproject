import os
import cv2
import numpy as np
import tensorflow as tf
from keras import models, layers
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

def load_regression_dataset(folder, label_file):
    images = []
    labels = []

    with open(label_file, 'r') as f:
        lines = f.readlines()

    for file, line in zip(os.listdir(folder), lines):
        if file.endswith(('.jpg', '.jpeg', '.png')):
            img_path = os.path.join(folder, file)
            image = cv2.imread(img_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (64, 64))
            images.append(image)

            # 获取回归标签
            label_parts = list(map(float, line.split()[1:]))
            labels.append(label_parts)

    return np.array(images), np.array(labels)

# 数据集路径
data_folder = "genki4kcutted"  # 数据集文件夹路径
label_file = "labels.txt"  # 包含标签信息的文本文件路径

# 加载回归数据集
X_reg, y_reg = load_regression_dataset(data_folder, label_file)

# 划分训练集、验证集和测试集（按60-20-20的百分比）
X_train, X_temp, y_train, y_temp = train_test_split(X_reg, y_reg, test_size=0.4, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# 预处理回归数据
X_train = X_train / 255.0  # 归一化到 [0, 1]
X_val = X_val / 255.0
X_test = X_test / 255.0
y_train = np.array(y_train)
y_val = np.array(y_val)
y_test = np.array(y_test)

# 构建回归模型
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(3, activation='linear'))  # 线性激活函数
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mse', 'mae', 'mape', 'cosine_similarity'])


# 使用早停法
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# 设置模型保存
model_checkpoint = ModelCheckpoint(filepath='best_model_convNext_re.h5', save_best_only=True)

# 训练回归模型
history = model.fit(X_train, y_train, epochs=40, batch_size=4, validation_data=(X_val, y_val),
                    callbacks=[early_stopping, model_checkpoint])

# 评估回归模型
y_pred_reg = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred_reg)
mae = np.mean(np.abs(y_test - y_pred_reg))


print(f'Mean Squared Error on Test Set: {mse}')
print(f'Mean Absolute Error on Test Set: {mae}')


# 打印图像和预测结果
for i in range(5):  # 打印前5个测试样本
    plt.figure(figsize=(8, 4))

    # 打印原始图像
    plt.subplot(1, 2, 1)
    plt.imshow(X_test[i])
    plt.title('Original Image')

    # 打印真实值和预测值
    plt.subplot(1, 2, 2)
    plt.scatter([0, 1, 2], y_test[i], label='True')
    plt.scatter([0, 1, 2], y_pred_reg[i], label='Predicted')
    plt.title(f'Regression Results - Sample {i + 1}')
    plt.legend()

    plt.show()
