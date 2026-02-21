---

# Aimonk Multilabel Image Classification

## Problem Statement

This project addresses a multi-label image classification task where each image may contain multiple attributes. 
The objective is to fine-tune a pretrained deep learning model to predict the presence or absence of four attributes while handling missing labels and class imbalance.

---

## Dataset Description

The dataset consists of 972 dress images and a corresponding `labels.txt` file containing four attribute annotations per image.

Each attribute is labeled as:

* `1` → Attribute present
* `0` → Attribute absent
* `NA` → Label information missing

Images with missing files were automatically removed during preprocessing.

---

## Approach

### 1. Data Preprocessing

* Parsed space-separated label file into a structured DataFrame.
* Converted `NA` values to `-1` to represent missing labels.
* Removed image paths that were not present in the dataset.
* Performed an 80-20 train-validation split.
* Built an efficient `tf.data` pipeline with batching, caching, and prefetching.

---

### 2. Handling Missing Labels

Since some attributes were marked as `NA`, these were converted to `-1` and excluded from loss computation using a masking mechanism.

During training:

* Labels equal to `-1` are ignored.
* Only valid labels contribute to the loss.

This ensures the model learns from partially labeled samples without discarding data.

---

### 3. Handling Class Imbalance

The dataset was highly imbalanced, especially for `attr4`.

Positive class weights were computed as:

```
pos_weight = (number of negative samples) / (number of positive samples)
```

These weights were incorporated into a custom weighted Binary Cross-Entropy loss to ensure rare positive attributes were not ignored during training.

---

### 4. Model Architecture

* Backbone: Pretrained ResNet50 (ImageNet weights)
* Global Average Pooling layer
* Dense layer with 4 output logits (one per attribute)

The backbone was frozen during training to prevent overfitting due to the small dataset size.

Binary Cross-Entropy with logits was used for multi-label learning.

---

## Training Details

* Optimizer: Adam
* Learning Rate: 1e-4
* Epochs: 5
* Custom Masked + Weighted BCE Loss

Training loss decreased steadily while validation loss remained relatively stable, indicating controlled learning without severe overfitting

Given the relatively small dataset size, the pretrained backbone was kept frozen to reduce the risk of overfitting. 
Early stopping after 5 epochs was chosen as validation loss began to plateau, indicating that further training could lead to overfitting.

---

## Loss Curve

The training loss curve is provided in `loss_curve.png`.

Axes:

* X-axis: iteration_number
* Y-axis: training_loss
* Title: Aimonk_multilabel_problem

---

## Inference

During inference:

* The model outputs logits.
* Sigmoid is applied to convert logits into probabilities.
* Attributes with probability > 0.5 are considered present.

Example output:

```
Predicted Attributes: ['attr2', 'attr3']
Probabilities: [0.43, 0.75, 0.67, 0.12]
```

---

## Reproducibility

The trained model file is not included due to GitHub file size limitations.

To reproduce results:

1. Place the dataset inside the project directory.
2. Install dependencies:

   ```
   pip install -r requirements.txt
   ```
3. Run the notebook to train and generate the model file.

---

## Key Highlights

* Multi-label classification
* Masked loss for partially labeled data
* Class imbalance handling using weighted BCE
* Pretrained fine-tuning
* Clean and modular TensorFlow pipeline

---

## Possible Improvements

Due to time constraints, advanced techniques such as data augmentation (random flips, rotations, color jitter), learning rate scheduling, early stopping, threshold tuning, and focal loss could further improve performance. These techniques can help improve generalization, especially for highly imbalanced attributes like attr4.

----

## Author

Rushda Aslami

MSc Data Science

GitHub: [https://github.com/rushda-code](https://github.com/rushda-code)

---
