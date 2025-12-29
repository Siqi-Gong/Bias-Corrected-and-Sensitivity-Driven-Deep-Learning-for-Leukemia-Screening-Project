# Microscopic Diagnosis of Leukemia using Deep Learning & Statistical Optimization
# author: Siqi Gong

import os
import argparse
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adamax
from tensorflow.keras import regularizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.metrics import Recall, Precision
from sklearn.utils import class_weight

from utils import preprocess, trim, LRA, tr_plot, print_info, saver, print_in_color

def parse_args():
    parser = argparse.ArgumentParser(description="Leukemia Diagnosis Pipeline")
    parser.add_argument("--data_dir", type=str, default="../input/leukemia-classification/C-NMC_Leukemia/training_data", 
                        help="Path to dataset directory")
    parser.add_argument("--working_dir", type=str, default="./results", help="Directory to save results")
    parser.add_argument("--epochs", type=int, default=40, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=40, help="Batch size")
    return parser.parse_args()

def main():
    args = parse_args()
    if not os.path.exists(args.working_dir):
        os.makedirs(args.working_dir)

    img_size = (300, 300)
    channels = 3
    img_shape = (img_size[0], img_size[1], channels)
    class_count = 2 # Normal vs Leukemia

    # 1. Cohort Construction (Stratified Sampling)
    # Biostats: Ensures consistent disease prevalence across splits to prevent distribution shift.
    try:
        train_df, test_df, valid_df = preprocess(args.data_dir, trsplit=0.9, vsplit=0.05)
    except Exception as e:
        print(f"Error loading data from {args.data_dir}: {e}")
        return

    # 2. Bias Mitigation (Under-sampling)
    # Biostats: Corrects Selection Bias by balancing majority class (HEM) to match minority (ALL).
    max_samples = 3000 # Limit majority class size
    min_samples = 0
    train_df = trim(train_df, max_samples, min_samples, 'labels')

    # 3. Data Augmentation
    # Biostats: Simulates biological variability (rotation, flip) to improve robustness.
    # Note: EfficientNet expects pixels in range 0-255, so no rescaling (1./255) is needed here if using 'imagenet' weights properly via internal preprocessing,
    # but the notebook used a scalar identity function. We keep it consistent.
    def scalar(img):
        return img 

    trgen = ImageDataGenerator(preprocessing_function=scalar, horizontal_flip=True, rotation_range=20)
    tvgen = ImageDataGenerator(preprocessing_function=scalar)

    train_gen = trgen.flow_from_dataframe(train_df, x_col='filepaths', y_col='labels', target_size=img_size, 
                                          class_mode='categorical', color_mode='rgb', shuffle=True, batch_size=args.batch_size)
    
    valid_gen = tvgen.flow_from_dataframe(valid_df, x_col='filepaths', y_col='labels', target_size=img_size, 
                                          class_mode='categorical', color_mode='rgb', shuffle=True, batch_size=args.batch_size)
    
    # Calculate optimal batch size for testing based on dataset length
    length = len(test_df)
    test_batch_size = sorted([int(length/n) for n in range(1, length+1) if length % n == 0 and length/n <= 80], reverse=True)[0]
    test_steps = int(length / test_batch_size)
    
    test_gen = tvgen.flow_from_dataframe(test_df, x_col='filepaths', y_col='labels', target_size=img_size, 
                                         class_mode='categorical', color_mode='rgb', shuffle=False, batch_size=test_batch_size)

    # 4. Model Definition (EfficientNetB3)
    # Biostats: High-dimensional feature extraction for unstructured image data.
    model_name = 'EfficientNetB3'
    base_model = tf.keras.applications.efficientnet.EfficientNetB3(
        include_top=False, weights="imagenet", input_shape=img_shape, pooling='max')
    
    x = base_model.output
    x = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)(x)
    x = Dense(256, kernel_regularizer=regularizers.l2(0.016),
              activity_regularizer=regularizers.l1(0.006),
              bias_regularizer=regularizers.l1(0.006), activation='relu')(x)
    x = Dropout(rate=0.45, seed=123)(x)
    output = Dense(class_count, activation='softmax')(x)
    
    model = Model(inputs=base_model.input, outputs=output)

    model.compile(Adamax(learning_rate=0.001), 
                  loss='categorical_crossentropy', 
                  metrics=['accuracy', Recall(name='recall'), Precision(name='precision')])
    
    # calculate Class Weights (cost0sensitive learning)
    train_labels = train_gen.classes 
    class_indices = train_gen.class_indices 
    print(f"Class Mapping: {class_indices}") 

    class_weights_vals = class_weight.compute_class_weight(
        class_weight='balanced',
        classes=np.unique(train_labels),
        y=train_labels
    )
    class_weights = dict(enumerate(class_weights_vals))

    target_class_name = 'ALL' 
    if target_class_name in class_indices:
        target_idx = class_indices[target_class_name]
        class_weights[target_idx] *= 2.0  
        print(f"Adjusted Class Weights (Penalizing Misses): {class_weights}")
    else:
        print("Warning: Target class name not found in indices, using balanced weights.")
        print(f"Balanced Weights: {class_weights}")

    # 5. Training with LRA
    callbacks = [LRA(model=model, base_model=base_model, patience=1, stop_patience=3, threshold=0.9,
                     factor=0.5, dwell=True, batches=len(train_gen), initial_epoch=0, epochs=args.epochs, ask_epoch=5)]

    print("Starting training...")

    history = model.fit(x=train_gen, 
                        epochs=args.epochs, 
                        verbose=1, 
                        callbacks=callbacks, 
                        validation_data=valid_gen, 
                        validation_steps=None, 
                        shuffle=False, 
                        initial_epoch=0,
                        class_weight=class_weights) 
    
    # 6. Evaluation & Visualization
    print("\nTraining complete. Plotting history...")
    tr_plot(history, 0)

    print("\nEvaluating on Test Set...")
    # Focus on Sensitivity (Recall) to minimize Type II errors (False Negatives).
    preds = model.predict(test_gen, verbose=1, steps=test_steps)
    acc = print_info(test_gen, preds, print_code=0, save_dir=args.working_dir, subject='leukemia')

    # 7. Save Model
    saver(args.working_dir, model, model_name, 'leukemia', acc, img_size, 1, train_gen)
    print(f"\n[Done] Model saved to {args.working_dir}")

if __name__ == "__main__":
    main()
