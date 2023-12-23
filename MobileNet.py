import os
import cv2
import numpy as np
import tensorflow as tf
from keras.applications import MobileNetV2
from keras import layers, models
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

def load_gender_dataset(folder, label_file, num_images=1000):
    images = []
    labels = []

    with open(label_file, 'r') as f:
        lines = f.readlines()

    for i, (file, line) in enumerate(zip(os.listdir(folder), lines)):
        if i >= num_images:
            break

        if file.endswith(('.jpg', '.jpeg', '.png')):
            img_path = os.path.join(folder, file)
            image = cv2.imread(img_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (224, 224))
            images.append(image)

            # 获取性别标签
            gender_label = int(line.split()[0])
            labels.append(gender_label)

    return np.array(images), np.array(labels)

# 数据集路径
data_folder = "genki4k/files"  # 数据集文件夹路径
label_file = "genderlabel.txt"  # 包含性别标签信息的文本文件路径

# 加载性别分类数据集的前1000张图像
X_gender, y_gender = load_gender_dataset(data_folder, label_file, num_images=1000)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_gender, y_gender, test_size=0.2, random_state=42)

# 数据增强
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# 预处理数据
X_train = X_train / 255.0
X_test = X_test / 255.0
y_train = tf.keras.utils.to_categorical(y_train, num_classes=2)
y_test = tf.keras.utils.to_categorical(y_test, num_classes=2)

# 构建MobileNetV2模型
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(2, activation='softmax')
])

# 编译模型
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 使用早停法
early_stopping = EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)

# 设置模型保存
model_checkpoint = ModelCheckpoint(filepath='best_model_gender_classification.h5', save_best_only=True)

# 训练模型
history = model.fit(datagen.flow(X_train, y_train, batch_size=32), epochs=10, validation_data=(X_test, y_test),
                    callbacks=[early_stopping, model_checkpoint])

# 评估模型
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test accuracy: {accuracy}')

# 进行性别预测
y_pred = model.predict(X_test)
y_pred_binary = np.argmax(y_pred, axis=1)

# 计算其他指标
precision = precision_score(np.argmax(y_test, axis=1), y_pred_binary)
recall = recall_score(np.argmax(y_test, axis=1), y_pred_binary)
f1 = f1_score(np.argmax(y_test, axis=1), y_pred_binary)

print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'F1 Score: {f1}')

# 计算混淆矩阵
conf_matrix = confusion_matrix(np.argmax(y_test, axis=1), y_pred_binary)
print('Confusion Matrix:')
print(conf_matrix)

# 使用 seaborn 绘制热力图
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=['Male', 'Female'], yticklabels=['Male', 'Female'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()
