import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models, callbacks
import matplotlib.pyplot as plt
import os

# Paths
train_dir = 'Fruit_classification/train'
test_dir = 'Fruit_classification/test'

# Data generators with augmentation
train_gen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=25,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    shear_range=0.15,
    horizontal_flip=True
).flow_from_directory(
    train_dir,
    target_size=(100, 100),
    batch_size=32,
    class_mode='categorical'
)

test_gen = ImageDataGenerator(rescale=1./255).flow_from_directory(
    test_dir,
    target_size=(100, 100),
    batch_size=32,
    class_mode='categorical'
)
# ---------------- Improved CNN Model ---------------- #
model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(100,100,3)),
    layers.MaxPooling2D(2,2),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),
    layers.Conv2D(128, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.4),
    layers.Dense(train_gen.num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()
checkpoint = callbacks.ModelCheckpoint(
    'best_fruit_model.h5', monitor='val_accuracy', save_best_only=True, mode='max', verbose=1
)

early_stop = callbacks.EarlyStopping(
    monitor='val_accuracy', patience=5, restore_best_weights=True, verbose=1
)
# ---------------- Train Model ---------------- #
history = model.fit(
    train_gen,
    validation_data=test_gen,
    epochs=20,
    callbacks=[checkpoint, early_stop]
)
# 📊 6. Plot accuracy/loss
# -----------------------
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.legend()
plt.title('Model Accuracy')

plt.subplot(1,2,2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.legend()
plt.title('Model Loss')
plt.show()

best_model = tf.keras.models.load_model('best_fruit_model.h5')
loss, acc = best_model.evaluate(test_gen)
print(f"\n✅ Best model accuracy on test data: {acc*100:.2f}%")


best_model.save("final_fruit_classifier.h5")
print("✅ Saved best model as 'best_fruit_model.h5' and 'final_fruit_classifier.h5'")