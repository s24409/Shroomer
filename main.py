import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
import pandas as pd
import os

# Paths to your image directory and metadata
train_metadata_path = '/path_to_train_metadata.csv'
train_image_dir = '/path_to_train_images/'
batch_size = 32
image_size = (224, 224)

# Load metadata
metadata = pd.read_csv(train_metadata_path)

# Create a dictionary mapping image paths to their class labels
image_paths = [os.path.join(train_image_dir, fname) for fname in metadata['image_path']]
labels = metadata['class_id'].values

# Create an ImageDataGenerator for data augmentation and normalization
datagen = ImageDataGenerator(
    rescale=1./255,  # Rescale pixel values between 0 and 1
    validation_split=0.2  # Use 20% of the data for validation
)

# Create training and validation datasets
train_generator = datagen.flow_from_dataframe(
    dataframe=metadata,
    directory=train_image_dir,
    x_col='image_path',
    y_col='class_id',
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'
)

validation_generator = datagen.flow_from_dataframe(
    dataframe=metadata,
    directory=train_image_dir,
    x_col='image_path',
    y_col='class_id',
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'
)

# Load the pre-trained MobileNetV2 model, without the top layers
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze the base model
base_model.trainable = False

# Create the final model
model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(1024, activation='relu'),
    Dense(train_generator.num_classes, activation='softmax')  # Number of mushroom classes
])

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=10
)

# Save the model
model.save('mushroom_classifier.h5')

# Evaluate the model
loss, accuracy = model.evaluate(validation_generator)
print(f'Validation accuracy: {accuracy:.2f}')
