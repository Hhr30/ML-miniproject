import os
import cv2
import numpy as np
import tensorflow as tf
from keras import models, layers
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

def load_dataset(folder, label_file):
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

            # 获取分类标签
            label = int(line.split()[0])
            labels.append(label)

    return np.array(images), np.array(labels)


# 数据集路径
data_folder = "genki4kcutted"  # 数据集文件夹路径
label_file = "labels.txt"  # 包含标签信息的文本文件路径

# 加载分类数据集
X, y = load_dataset(data_folder, label_file)

# 划分训练集、验证集和测试集（按60-20-20的百分比）
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# 预处理分类数据
X_train = X_train / 255.0  # 归一化到 [0, 1]
X_val = X_val / 255.0
X_test = X_test / 255.0
y_train = tf.keras.utils.to_categorical(y_train, num_classes=2)  # 独热编码
y_val = tf.keras.utils.to_categorical(y_val, num_classes=2)
y_test = tf.keras.utils.to_categorical(y_test, num_classes=2)

# 构建 ConvNeXt 模型
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
model.add(layers.Dense(2, activation='sigmoid'))
model.summary()


# 使用SGD优化器,设置学习率
sgd_optimizer = tf.keras.optimizers.SGD(learning_rate=0.001, momentum=0.9)

model.compile(optimizer=sgd_optimizer, loss='binary_crossentropy', metrics=['accuracy'])

# 使用早停法
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# 设置模型保存
model_checkpoint = ModelCheckpoint(filepath='best_model_convNext.h5', save_best_only=True)

# 训练模型
model.fit(X_train, y_train, epochs=40, batch_size=4, validation_data=(X_val, y_val), callbacks=[early_stopping, model_checkpoint])

# 评估分类模型
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test accuracy: {accuracy}')

# 进行分类预测
y_pred = model.predict(X_test)
y_pred_binary = np.argmax(y_pred, axis=1)

# 计算分类的其他指标
precision = precision_score(np.argmax(y_test, axis=1), y_pred_binary)
recall = recall_score(np.argmax(y_test, axis=1), y_pred_binary)
f1 = f1_score(np.argmax(y_test, axis=1), y_pred_binary)

print(f'Classification Precision: {precision}')
print(f'Classification Recall: {recall}')
print(f'Classification F1 Score: {f1}')

# 计算分类混淆矩阵
conf_matrix = confusion_matrix(np.argmax(y_test, axis=1), y_pred_binary)
print('Classification Confusion Matrix:')
print(conf_matrix)

# 使用 seaborn 绘制热力图
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=['Non-Smile', 'Smile'], yticklabels=['Non-Smile', 'Smile'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()
