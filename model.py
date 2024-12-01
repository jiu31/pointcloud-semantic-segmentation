import tensorflow as tf

class MatMulLayer(tf.keras.layers.Layer):
    """
    Keras 레이어로 정의된 tf.linalg.matmul
    """
    def call(self, inputs):
        x, transformation_matrix = inputs
        return tf.linalg.matmul(x, transformation_matrix)

def transformation_block(inputs, num_features):
    """
    Transformation Network
    Args:
        inputs: 입력 텐서
        num_features: 변환 행렬 크기 (num_features x num_features)
    Returns:
        변환 행렬 텐서
    """
    x = tf.keras.layers.Conv1D(64, kernel_size=1, activation="relu")(inputs)
    x = tf.keras.layers.Conv1D(128, kernel_size=1, activation="relu")(x)
    x = tf.keras.layers.Conv1D(1024, kernel_size=1, activation="relu")(x)
    x = tf.keras.layers.GlobalMaxPooling1D()(x)

    # Fully connected layers for the transformation matrix
    x = tf.keras.layers.Dense(512, activation="relu")(x)
    x = tf.keras.layers.Dense(256, activation="relu")(x)
    x = tf.keras.layers.Dense(num_features * num_features, activation="linear", kernel_initializer="zeros")(x)
    transformation_matrix = tf.keras.layers.Reshape((num_features, num_features))(x)

    return transformation_matrix

def create_pointnet_model(num_sample_points, num_classes):
    """
    PointNet 모델 정의

    Returns:
        PointNet 모델
    """
    inputs = tf.keras.Input(shape=(num_sample_points, 3))  # (batch_size, num_points, 3)

    # Input transformation
    transformation_matrix = transformation_block(inputs, num_features=3)
    transformed_inputs = MatMulLayer()([inputs, transformation_matrix])

    # Feature extraction
    x = tf.keras.layers.Conv1D(64, kernel_size=1, activation="relu")(transformed_inputs)
    x = tf.keras.layers.Conv1D(64, kernel_size=1, activation="relu")(x)

    # Feature transformation
    feature_transformation_matrix = transformation_block(x, num_features=64)
    transformed_features = MatMulLayer()([x, feature_transformation_matrix])

    # Global feature extraction
    x = tf.keras.layers.Conv1D(64, kernel_size=1, activation="relu")(transformed_features)
    x = tf.keras.layers.Conv1D(128, kernel_size=1, activation="relu")(x)
    x = tf.keras.layers.Conv1D(1024, kernel_size=1, activation="relu")(x)
    global_features = tf.keras.layers.GlobalMaxPooling1D()(x)  # (batch_size, 1024)

    # Duplicate global features for each point
    global_features = tf.keras.layers.RepeatVector(num_sample_points)(global_features)
    global_features = tf.keras.layers.Concatenate()([transformed_features, global_features])

    # Point-wise classification
    x = tf.keras.layers.Conv1D(512, kernel_size=1, activation="relu")(global_features)
    x = tf.keras.layers.Conv1D(256, kernel_size=1, activation="relu")(x)
    outputs = tf.keras.layers.Conv1D(num_classes, kernel_size=1, activation="softmax")(x)

    return tf.keras.Model(inputs, outputs)