

import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import Xception



#  Attention Modules


def channel_attention_block(x, ratio: int = 8):
    """
    Channel Attention: squeeze feature channels with shared MLP.
    Uses sigmoid (not softmax) so multiple channels can be
    activated independently — suits multi-region face artifacts.
    """
    channels = x.shape[-1]

    # Shared MLP weights
    fc1 = layers.Dense(channels // ratio, activation='relu', use_bias=False)
    fc2 = layers.Dense(channels, use_bias=False)

    # Average-pool branch
    avg = layers.GlobalAveragePooling2D()(x)
    avg = layers.Reshape((1, 1, channels))(avg)
    avg_out = fc2(fc1(avg))

    # Max-pool branch
    max_ = layers.GlobalMaxPooling2D()(x)
    max_ = layers.Reshape((1, 1, channels))(max_)
    max_out = fc2(fc1(max_))

    # Sigmoid gate
    scale = layers.Activation('sigmoid')(avg_out + max_out)
    return layers.Multiply()([x, scale])


def spatial_attention_block(x, kernel_size: int = 7):
    """
    Spatial Attention: highlights discriminative positions
    (eyes, mouth edges, hairline) across the feature map.
    """
    # Pool along channel axis
    avg_pool = tf.reduce_mean(x, axis=-1, keepdims=True)
    max_pool = tf.reduce_max(x,  axis=-1, keepdims=True)

    concat = layers.Concatenate(axis=-1)([avg_pool, max_pool])

    # Single conv → sigmoid spatial mask
    scale = layers.Conv2D(
        1, kernel_size, padding='same',
        activation='sigmoid', use_bias=False
    )(concat)
    return layers.Multiply()([x, scale])



#  Full Model


def build_model(input_shape=(224, 224, 3), freeze_backbone: bool = False):
    """
    Build the Hybrid CNN-Attention model.

    Pipeline (matches Table 2 of the paper):
      Input (224×224×3)
        → Xception pretrained backbone  → Feature Maps (7×7×2048)
        → Channel Attention
        → Spatial Attention
        → Global Average Pooling (2048)
        → Dense 512 + ReLU
        → Dropout 0.5
        → Dense 1 + Sigmoid  (Real=0 / Fake=1)

    Args:
        input_shape:      (H, W, C) — default 224×224×3
        freeze_backbone:  if True, Xception weights are frozen
                          (useful for fast demo / transfer-only mode)
    """
    # Xception backbone 
    backbone = Xception(
        include_top=False,
        weights='imagenet',
        input_shape=input_shape
    )
    backbone.trainable = not freeze_backbone  # fine-tune end-to-end

    inputs = backbone.input                   # 224×224×3
    x      = backbone.output                  # 7×7×2048

    # Dual Attention 
    x = channel_attention_block(x, ratio=8)
    x = spatial_attention_block(x, kernel_size=7)

    # Classifier head 
    x = layers.GlobalAveragePooling2D()(x)    # 2048
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(1, activation='sigmoid', name='output')(x)

    model = Model(inputs=inputs, outputs=outputs,
                  name='HybridCNN_Attention_DeepfakeDetector')
    return model


if __name__ == '__main__':
    model = build_model()
    model.summary()
    print(f"\nTotal params      : {model.count_params():,}")
    print(f"Trainable params  : {sum(tf.size(w).numpy() for w in model.trainable_weights):,}")
