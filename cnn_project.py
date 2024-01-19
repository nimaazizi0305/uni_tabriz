import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator


dataset_path_train='D:\\master_of_uni_tabriz\\ai_project\Traffic sign dataset\\train'
dataset_path_test='D:\\master_of_uni_tabriz\\ai_project\Traffic sign dataset\\test'

# تعیین مسیر دیتاست

# ایجاد یک مولد داده برای آموزش
train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

# تعیین مسیر داده‌های آموزش و اعتبارسنجی
train_generator = train_datagen.flow_from_directory(
    dataset_path_train,
    target_size=(224, 224),  # اندازه تصاویر
    batch_size=32,
    class_mode='categorical',
    subset='training'  # تقسیم 80% برای آموزش
)

validation_generator = train_datagen.flow_from_directory(
    dataset_path_test,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='validation'  # تقسیم 20% برای اعتبارسنجی
)

num_classes = 15  # به عنوان یک مثال، تعداد کلاس‌های موجود در دیتاست


# ساخت مدل CNN
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(num_classes, activation='softmax'))

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])


model.fit(train_generator, epochs=10, validation_data=validation_generator)

model.save('traffic_signs_model.h5')

history = model.fit(train_generator, epochs=10, validation_data=validation_generator)

training_accuracy = history.history['accuracy']
validation_accuracy = history.history['val_accuracy']
