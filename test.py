import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pandas as pd

# Path to your saved model
model_path = 'mushroom_classifier.h5'

# Load the trained model
model = load_model(model_path)

# Paths to your test metadata and image directory
test_metadata_path = 'DF20-public_test_metadata_PROD-2.csv'
test_image_dir = 'DF20_300'  # Assuming the images are in the same directory
batch_size = 32
image_size = (224, 224)

# Load test metadata
test_metadata = pd.read_csv(test_metadata_path)

# Convert 'class_id' column to string (required for categorical mode)
test_metadata['class_id'] = test_metadata['class_id'].astype(str)

# Create an ImageDataGenerator for normalization (rescale)
datagen = ImageDataGenerator(rescale=1./255)

# Create test dataset
test_generator = datagen.flow_from_dataframe(
    dataframe=test_metadata,
    directory=test_image_dir,
    x_col='image_path',
    y_col='class_id',
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False  # No need to shuffle for test evaluation
)

# Evaluate the model on the test data
loss, accuracy = model.evaluate(test_generator)
print(f'Test accuracy: {accuracy:.2f}, Test loss: {loss:.2f}')
