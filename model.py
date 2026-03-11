import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import Xception


def channel_attention(x, ratio=8):
    ch = x.shape[-1]

    # shared weights between avg and max branches
    fc1 = layers.Dense(ch // ratio, activation='relu', use_bias=False)
    fc2 = layers.Dense(ch, use_bias=False)

    avg = layers.Reshape((1, 1, ch))(layers.GlobalAveragePooling2D()(x))
    mx  = layers.Reshape((1, 1, ch))(layers.GlobalMaxPooling2D()(x))

    # sigmoid not softmax — with softmax the model was competing between
    # channels which hurt on faces with multiple artifact regions at once
    out = layers.Activation('sigmoid')(fc2(fc1(avg)) + fc2(fc1(mx)))
    return layers.Multiply()([x, out])


def spatial_attention(x, k=7):
    avg  = tf.reduce_mean(x, axis=-1, keepdims=True)
    mx   = tf.reduce_max(x,  axis=-1, keepdims=True)
    cat  = layers.Concatenate(axis=-1)([avg, mx])
    mask = layers.Conv2D(1, k, padding='same', activation='sigmoid', use_bias=False)(cat)
    return layers.Multiply()([x, mask])


def build_model(input_shape=(224, 224, 3), freeze_backbone=False):
    # went with Xception over ResNet50 — depthwise separable convs pick up
    # texture artifacts more cleanly, and it was already fine-tuned well on faces
    backbone = Xception(include_top=False, weights='imagenet', input_shape=input_shape)
    backbone.trainable = not freeze_backbone

    x = backbone.output  # 7x7x2048 after exit flow

    x = channel_attention(x)
    x = spatial_attention(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    out = layers.Dense(1, activation='sigmoid', name='output')(x)

    return Model(backbone.input, out, name='deepfake_detector')


if __name__ == '__main__':
    m = build_model()
    m.summary()
