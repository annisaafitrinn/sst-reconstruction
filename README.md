**Sea Surface Temperature Reconstruction**

This repository contains code for a project that reconstructs sea surface temperature in occluded regions of satellite images using a U-Net architecture. The model focuses on accurately predicting missing parts of the images by utilizing a custom loss function tailored for these areas, optimizing for Root Mean Square Error (RMSE). Our objective is to improve upon a statistical baseline model with an RMSE of 0.6, aiming to achieve meaningful enhancements through deep learning techniques.

**Model Architecture**

The model follows a U-Net architecture, which is particularly effective in segmentation and reconstruction tasks. It consists of an encoder-decoder structure:

Encoder: Extracts high-level features using convolutional layers, batch normalization, and dropout for regularization.
Bottleneck: Deeper layers to capture complex features.
Decoder: Reconstructs the image by progressively upsampling and concatenating encoder features via skip connections.

The output has 3 channels:

- The predicted image.
- The predicted real mask.
- The difference mask.
- Custom Loss Function

The model uses a custom loss function based on Root Mean Squared Error (RMSE), which focuses on predicting the occluded regions of the image. Specifically, the first channel of y_true and y_pred (the original and predicted images) are compared only in the areas indicated by an artificial mask (third channel of y_true), guiding the model to learn and improve predictions in these regions.

Note: The RMSE monitored during training is not denormalized, and further denormalization will be required during evaluation. However, monitoring this value helps track model progress.

**Dataset**

The data used in this project is from the MODIS dataset, specifically nightly data collected by the Aqua satellite. The dataset provides valuable insights for temperature estimation tasks, particularly in challenging or occluded regions

The data is split into input (batch_x) and output (batch_y) batches:

batch_x contains:
The masked image.
The artificial mask.
The land-sea mask.
The tuned baseline.
batch_y contains:
The actual image.
The real mask.
The difference mask.

**Training**

The model is trained for 150 epochs with a batch size of 32. We use the following callbacks:

ModelCheckpoint: Saves the best model based on validation loss.
EarlyStopping: Stops training if validation loss does not improve for 10 epochs.
ReduceLROnPlateau: Reduces the learning rate when the model's performance plateaus.

python train.py

**Dependencies**

TensorFlow/Keras
NumPy
Matplotlib

**Results**

During training, we monitor the RMSE for both the training and validation sets. Note that these RMSE values are computed before denormalization. Final evaluation will require the denormalized RMSE values. We reached RMSE of 0.48, which is slightly better result the RMSE given by the statistical baseline (0.63).

Improvement

Another model can be explored to achieve better result, such as using Visual Transformer.

**Acknowledgment**

This model was inspired by the work of Wang et al. (2024), who demonstrated the effectiveness of U-Net for occlusion prediction tasks in remote sensing images.

Wang, C., Sun, L., Huang, B., Zhang, D., Mu, J., & Wu, J. (2024). Title of the paper. Remote Sensing, 16(7), 1205. https://doi.org/10.3390/rs16071205

**License**

This project is licensed under the MIT License - see the LICENSE file for details.
