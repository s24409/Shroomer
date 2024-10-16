import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
import pandas as pd
import os
from sklearn.model_selection import train_test_split

# Paths to your image directory and metadata
train_metadata_path = 'DF20-train_metadata_PROD-2.csv'
train_image_dir = 'DF20_300'
batch_size = 32
image_size = (224, 224)

# Load metadata
metadata = pd.read_csv(train_metadata_path)

# Convert 'class_id' column to string (required for categorical mode)
metadata['class_id'] = metadata['class_id'].astype(str)

# Shuffle and split the dataset into training and validation sets (80% training, 20% validation)
train_metadata, val_metadata = train_test_split(metadata, test_size=0.2, random_state=42)

# Create an ImageDataGenerator for data augmentation and normalization
datagen = ImageDataGenerator(rescale=1./255)

# Create training and validation datasets using the split data
train_generator = datagen.flow_from_dataframe(
    dataframe=train_metadata,
    directory=train_image_dir,
    x_col='image_path',
    y_col='class_id',
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical'
)

validation_generator = datagen.flow_from_dataframe(
    dataframe=val_metadata,
    directory=train_image_dir,
    x_col='image_path',
    y_col='class_id',
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical'
)

# Get the number of unique classes
num_classes = metadata['class_id'].nunique()

# Load the pre-trained MobileNetV2 model, without the top layers
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze the base model
base_model.trainable = False

# Create the final model
model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(1024, activation='relu'),
    Dense(num_classes, activation='softmax')  # Number of mushroom classes
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
