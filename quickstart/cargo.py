import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
# 1. 載入數據集
fashion_mnist = tf.keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# 定義類別名稱
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# 2. 數據預處理
# 歸一化
train_images = train_images / 255.0
test_images = test_images / 255.0

# 3. 創建數據增強層
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomZoom(0.1),
])

# 4. 構建優化後的模型
model = tf.keras.Sequential([
    data_augmentation,
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),  # 從 256 改為 128
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(64, activation='relu'),   # 從 128 改為 64
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10)
])

# 5. 設置學習率調度
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=1e-3,
    decay_steps=10000,
    decay_rate=0.9
)
optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

# 6. 編譯模型
model.compile(
    optimizer=optimizer,
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

# 7. 設置早停機制
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=3,
    restore_best_weights=True
)

# 8. 訓練模型
history = model.fit(
    train_images, 
    train_labels, 
    epochs=10,
    validation_split=0.2,
    callbacks=[early_stopping],
    batch_size=16  # 從 32 改為 16
)

# 9. 評估模型
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print('\nTest accuracy:', test_acc)

# 10. 創建預測模型
probability_model = tf.keras.Sequential([
    model,
    tf.keras.layers.Softmax()
])

# 11. 進行預測
predictions = probability_model.predict(test_images)

# 12. 繪圖函數定義
def plot_image(i, predictions_array, true_label, img):
    true_label, img = true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img, cmap=plt.cm.binary)

    predicted_label = np.argmax(predictions_array)
    color = 'blue' if predicted_label == true_label else 'red'

    plt.xlabel("{} {:2.0f}% ({})".format(
        class_names[predicted_label],
        100*np.max(predictions_array),
        class_names[true_label]
    ), color=color)

def plot_value_array(i, predictions_array, true_label):
    true_label = true_label[i]
    plt.grid(False)
    plt.xticks(range(10))
    plt.yticks([])
    thisplot = plt.bar(range(10), predictions_array, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)

    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')

# 13. 顯示訓練歷史
plt.figure(figsize=(12, 4))

# 繪製準確率
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# 繪製損失
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()

# 14. 顯示預測結果
num_rows = 5
num_cols = 3
num_images = num_rows * num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
    plt.subplot(num_rows, 2*num_cols, 2*i+1)
    plot_image(i, predictions[i], test_labels, test_images)
    plt.subplot(num_rows, 2*num_cols, 2*i+2)
    plot_value_array(i, predictions[i], test_labels)
plt.tight_layout()
plt.show()

# 15. 混淆矩陣
from sklearn.metrics import confusion_matrix
import seaborn as sns

# 獲取預測標籤
predicted_labels = np.argmax(predictions, axis=1)

# 計算混淆矩陣
cm = confusion_matrix(test_labels, predicted_labels)

# 繪製混淆矩陣
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.xticks(rotation=45)
plt.yticks(rotation=45)
plt.tight_layout()
plt.show()

# 16. 單一圖片預測示例
# 選擇一張測試圖片
img = test_images[1]
img_array = tf.keras.preprocessing.image.img_to_array(img)
img_array = tf.expand_dims(img_array, 0)

# 進行預測
predictions_single = probability_model.predict(img_array)

# 顯示預測結果
plt.figure(figsize=(6, 3))
plt.subplot(1, 2, 1)
plt.imshow(img, cmap=plt.cm.binary)
plt.title(f'True: {class_names[test_labels[1]]}')
plt.subplot(1, 2, 2)
plot_value_array(1, predictions_single[0], test_labels)
plt.title(f'Predicted: {class_names[np.argmax(predictions_single[0])]}')
plt.tight_layout()
plt.show()

# 17. 打印模型摘要
model.summary()

# 18. 打印分類報告
from sklearn.metrics import classification_report
print("\nClassification Report:")
print(classification_report(test_labels, predicted_labels, target_names=class_names))