# Wildfire Detection from Satellite Imagery (DS340W Final Project)

This repository contains the final code for a **DS340W** course project on
detecting wildfires from satellite imagery using **Convolutional Neural Networks (CNNs)**
and an extended **ConvLSTM** model.
The goal is to classify each satellite image into one of two classes:
- **Wildfire**
- **No Wildfire**
The dataset (originally used on Kaggle) consists of ~**42,850** RGB images of size
**350×350**, split into:
- **Wildfire:** 22,710 images  
- **No Wildfire:** 20,140 images  
To facilitate training, the dataset is split into:
- **Train:** ~70%
- **Validation:** ~15%
- **Test:** ~15%
---

## 1. Project Overview

The project started from an original CNN-based Kaggle notebook for wildfire
classification. On top of that baseline, this work:

1. Implements several **CNN baselines** (with and without data augmentation).  
2. Adds a **ConvLSTM** model to better capture spatial–temporal structure.  
3. Improves the **training pipeline** (augmentation, class weights, early stopping).  
4. Handles **corrupted images** more robustly.  
5. Performs **model comparison and error analysis** using confusion matrices and
   classification reports.

Although the current dataset provides only single images (no explicit time series),
the ConvLSTM pipeline is designed so that it can easily be extended to multi–time-step
inputs in future work.

---

## 2. Dataset

The notebook assumes the Kaggle directory structure:

```text
wildfire-prediction-dataset/
├── train/
│   ├── wildfire/
│   └── nowildfire/
├── valid/
│   ├── wildfire/
│   └── nowildfire/
└── test/
    ├── wildfire/
    └── nowildfire/
``` 
In the code, this is referenced as:

BASE_DIR  = "/kaggle/input/wildfire-prediction-dataset"
train_dir = os.path.join(BASE_DIR, "train")
val_dir   = os.path.join(BASE_DIR, "valid")
test_dir  = os.path.join(BASE_DIR, "test")
If you run the notebook outside Kaggle, you should:

Download the same dataset (from Kaggle).
（Place it in some local folder (data/wildfire-prediction-dataset).）
Update BASE_DIR (and the earlier dir, train_dir, val_dir, test_dir variables)
to point to your local path.
Each split has two subfolders:
wildfire/ → label 1 for wildfire prediction 
nowildfire/ → label 0 for true fire

## 3. Model
3.1 Baseline CNN (Model 1)

A relatively shallow CNN used as a baseline:

Input size: 128×128×3

Stacked Conv2D → BatchNorm → MaxPool blocks

L2 regularization and Dropout in deeper layers

Final Dense(1, activation="sigmoid") for binary classification

Loss: binary_crossentropy

Optimizer: Adam(learning_rate=1e-4, decay=1e-5)

Metrics: accuracy, recall

This model is trained using ImageDataGenerator with simple rescaling.

3.2 Augmented / Stronger CNNs (Model 2 & Model 3)

On top of the baseline, the notebook includes more advanced CNN variants
(e.g., deeper architectures and/or heavier augmentation).
They are trained on augmented image generators and evaluated on the same test set.

Key ideas:

Use on-the-fly data augmentation for robustness:

random flips, rotations, zooms, etc.

Encourage better generalization and reduce overfitting compared to the simple baseline.

3.3 ConvLSTM Model (Our Extension)

The lower half of the notebook (marked with:

# All the code below was written for ConvLSTM by us for the DS340W project.

implements a ConvLSTM-based network.

Main design:

Input size: 96×96×3

TIME_STEPS = 1 (current dataset has only one frame per sample)
but the model is defined for (time, H, W, C) and can be extended to TIME_STEPS > 1.

Two stacked ConvLSTM2D layers with BatchNormalization

A spatial Conv2D + MaxPooling2D block on top of the ConvLSTM output

Fully connected head:

Flatten → Dropout(0.4) → Dense(128, relu) → Dense(1, sigmoid)
Also, The ConvLSTM is trained using "tf.data" pipelines and supports
GPU acceleration.

# 4. Data Pipeline & Training
4.1 CNN Pipeline

For the initial CNN experiments, the notebook uses:

ImageDataGenerator(rescale=1./255.)

flow_from_directory(...) on train_dir, val_dir, test_dir

Batch size: BATCH_SIZE = 32

Target size: TARGET_SIZE = (128, 128)

The pipeline returns generators for:

train_generator

val_generator

test_generator (+ an augmented test generator for some models)

Standard model.fit(...) calls are used.

4.2 ConvLSTM Pipeline

For the ConvLSTM experiments, we switch to Keras directory loaders and tf.data:

Load datasets via tf.keras.utils.image_dataset_from_directory(...)

Apply:

Resizing to IMG_SIZE = (96, 96)

Data augmentation and normalization using tf.data.map

Caching and prefetching for better I/O performance

Add a time dimension:
```text
def add_time_dim(images, labels):
    # (batch, H, W, C) -> (batch, TIME_STEPS, H, W, C)
    images = tf.expand_dims(images, axis=1)
    return images, labels
```

Compute class weights by counting files in the train/ folders
(ignoring hidden files) to cope with class imbalance:
```text
class_weight = {
    0: total / (2.0 * neg),  # nowildfire
    1: total / (2.0 * pos),  # wildfire
}
```

Train with early stopping:
```text
history = model.fit(
    train_ds,
    epochs=EPOCHS,              # e.g., 5
    steps_per_epoch=TRAIN_STEPS, # e.g., 200
    validation_data=val_ds,
    validation_steps=VAL_STEPS,  # e.g., 50
    class_weight=class_weight,
    callbacks=[
        keras.callbacks.EarlyStopping(
            patience=2,
            restore_best_weights=True
        )
    ],
)
```

This setup is designed to be robust to bad / partially corrupted JPEGs and
to reduce overfitting.
## 5. Evaluation & Results
5.1 CNN Results (Original Baseline vs Stronger CNN)

Using the test generator (test_generator / test_generator_augm), the notebook
computes confusion matrices and prints classification reports.

One example result for the original CNN:

Accuracy: ~0.96

nowildfire: precision ≈ 0.96, recall ≈ 0.96

wildfire: precision ≈ 0.97, recall ≈ 0.97

A stronger CNN variant (Model 3) gives:

Accuracy: ~0.95

Slightly different precision/recall trade-offs across the two classes.

5.2 ConvLSTM Results

For the ConvLSTM model, we:

Sweep thresholds from 0.3 to 0.7 on the validation set to find the
best decision threshold (best_th) by accuracy.

Apply the best threshold to the test set, collecting predictions and
computing the confusion matrix:
```text
Confusion matrix (test, with best_th):
[[2746   74]
 [ 324 3124]]
```
Interpreting this:

TN (nowildfire correctly predicted): 2746

FP (nowildfire misclassified as wildfire): 74

FN (missed wildfires): 324

TP (wildfires correctly detected): 3124

From this we obtain approximately:

Accuracy: ~0.94

Wildfire precision: ~0.98

Wildfire recall: ~0.91

Nowildfire precision: ~0.89

Nowildfire recall: ~0.97
So we can compared to the strongest CNN, the ConvLSTM (with the chosen threshold)

# 6. How to Run
6.1 Requirements

Python 3.8+
Jupyter (or JupyterLab)
Libraries (see imports at the top of the notebook):
numpy,pandas,matplotlib,seaborn,opencv-python (cv2),Pillow,tensorflow / keras,albumentations,scikit-learn

6.2 Steps (Kaggle)

Create a Kaggle Notebook.

Add the wildfire-prediction-dataset dataset to the notebook.

Upload ds340w-wildfire Final code.ipynb (or copy the cells).

Run all cells from top to bottom (GPU accelerator recommended).
and all down!
# 7. File Structure

Currently the main file in this repo is:.
├── data/           # wildfire-prediction-dataset
├── notebooks/
│   └── ds340w-wildfire Final code.ipynb
└── README.md

# 8. Limitations & Future Work
The dataset currently provides single images, so ConvLSTM is used with
TIME_STEPS = 1. In the future, real multi-time-step sequences (may be use daily
satellite images) would allow ConvLSTM to fully exploit spatio-temporal dynamics.

# Thank You!
