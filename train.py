import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dropout, Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import os
import numpy as np
from collections import Counter

# Paths
train_dir = 'dataset/train'
val_dir = 'dataset/validation'

IMG_SIZE = 224
BATCH_SIZE = 16
NUM_CLASSES = 2
EPOCHS_PHASE1 = 10
EPOCHS_PHASE2 = 20

# Data augmentation for training
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.15,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    horizontal_flip=True,
    fill_mode="nearest"
)

val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=True
)

val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

# Calculate class weights to handle imbalance
counter = Counter(train_generator.classes)
max_count = float(max(counter.values()))
class_weights = {cls: max_count / count for cls, count in counter.items()}
print("Class weights:", class_weights)

# Build model with MobileNetV2 base
input_layer = Input(shape=(IMG_SIZE, IMG_SIZE, 3))
base_model = MobileNetV2(include_top=False, input_tensor=input_layer, weights='imagenet')

# Add classification head
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.5)(x)
output_layer = Dense(NUM_CLASSES, activation='softmax')(x)

model = Model(inputs=input_layer, outputs=output_layer)

# Phase 1: Freeze base model
base_model.trainable = False

model.compile(
    optimizer=Adam(learning_rate=1e-3),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

callbacks = [
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1),
    ModelCheckpoint('best_model_phase1.h5', monitor='val_loss', save_best_only=True, verbose=1)
]

print("Training phase 1 (top layers)...")
model.fit(
    train_generator,
    epochs=EPOCHS_PHASE1,
    validation_data=val_generator,
    class_weight=class_weights,
    callbacks=callbacks
)

# Phase 2: Unfreeze all layers and fine-tune
base_model.trainable = True

# Optionally, freeze some early layers to avoid overfitting
for layer in base_model.layers[:100]:  # freeze first 100 layers (adjust as needed)
    layer.trainable = False

model.compile(
    optimizer=Adam(learning_rate=1e-5),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

callbacks = [
    EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, verbose=1),
    ModelCheckpoint('best_model_final.h5', monitor='val_loss', save_best_only=True, verbose=1)
]

print("Training phase 2 (fine-tuning)...")
model.fit(
    train_generator,
    epochs=EPOCHS_PHASE2,
    validation_data=val_generator,
    class_weight=class_weights,
    callbacks=callbacks
)

print("Training complete. Final model saved as best_model_final.h5")
