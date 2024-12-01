import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from preprocess import load_and_preprocess
from model import create_pointnet_model
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils.class_weight import compute_class_weight
import os

# 하이퍼파라미터 정의
NUM_SAMPLE_POINTS = 2048
NUM_CLASSES = 10
BATCH_SIZE = 32
EPOCHS = 60
LEARNING_RATE = 0.001
GRID_SIZE = 10.0
MODEL_SAVE_PATH = "./pointnet_model.h5"

LABELS = [
    "Man-made terrain", "Natural terrain",
    "High vegetation", "Low vegetation", "Buildings",
    "Hard scape", "Scanning artifacts", "Cars"
]
COLORS = [
    [255, 0, 0],      # Man-made terrain - Red
    [0, 255, 0],      # Natural terrain - Green
    [0, 0, 255],      # High vegetation - Blue
    [255, 255, 0],    # Low vegetation - Yellow
    [255, 165, 0],    # Buildings - Orange
    [128, 0, 128],    # Hard scape - Purple
    [0, 255, 255],    # Scanning artifacts - Cyan
    [255, 105, 180]   # Cars - Hot Pink
]

# 데이터 파일 경로
POINTS_FILE = "./bildstein_station1_xyz_intensity_rgb.txt"
LABELS_FILE = "./bildstein_station1_xyz_intensity_rgb.labels"

def train_and_evaluate():
    """
    PointNet 모델 학습 및 평가 수행
    """
    # 데이터 로드 및 전처리
    print("Loading and preprocessing data...")
    oversampled_train_points, oversampled_train_labels, x_test, y_test = load_and_preprocess(
        points_file=POINTS_FILE,
        labels_file=LABELS_FILE,
        grid_size=GRID_SIZE,
        num_sample_points=NUM_SAMPLE_POINTS,
        num_classes=NUM_CLASSES
    )

    # 클래스 가중치 계산
    class_weights = calculate_class_weights(oversampled_train_labels, NUM_CLASSES)

    # PointNet 모델 생성
    print("Creating PointNet model...")
    model = create_pointnet_model(num_sample_points=NUM_SAMPLE_POINTS, num_classes=NUM_CLASSES)

    # 모델 구조 출력
    model.summary()

    # 모델 컴파일
    model.compile(
        loss="categorical_crossentropy",
        optimizer=Adam(learning_rate=LEARNING_RATE),
        metrics=["accuracy"]
    )

    # 모델 학습
    print("Starting training...")
    history = model.fit(
        oversampled_train_points,
        oversampled_train_labels,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=(x_test, y_test),
        class_weight=class_weights  # 클래스 가중치 적용
    )

    # 모델 저장
    print(f"Saving model to {MODEL_SAVE_PATH}...")
    model.save(MODEL_SAVE_PATH)

    # 학습 결과 시각화
    plot_training_history(history)

    # 모델 평가
    print("Evaluating model...")
    loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
    print(f"Test accuracy: {accuracy:.2f}")

    # 세그먼테이션 시각화
    print("Visualizing segmentation results...")
    visualize_segmentation_results(model, x_test, y_test)


def visualize_segmentation_results(model, x_test, y_test):
    """
    테스트 세그먼테이션 결과를 시각화합니다.

    Args:
        model: 학습된 PointNet 모델.
        x_test (numpy.ndarray): 테스트 데이터 포인트.
        y_test (numpy.ndarray): 테스트 데이터 레이블.
    """
    test_idx = 0  # 테스트 샘플 인덱스
    if test_idx >= len(x_test):
        print("Invalid test index. Skipping visualization.")
        return

    test_points = x_test[test_idx]
    test_true_labels = np.argmax(y_test[test_idx], axis=-1)
    test_predicted_labels = np.argmax(model.predict(test_points[None, ...]), axis=-1)[0]

    visualize_segmentation_with_legend(
        test_points,
        test_true_labels,
        test_predicted_labels,
        num_sample_points=NUM_SAMPLE_POINTS,
        labels=LABELS,
        colors=COLORS
    )


def plot_training_history(history):
    """
    학습 결과를 시각화합니다.

    Args:
        history: 모델 학습 기록 (History 객체).
    """
    plt.figure(figsize=(12, 5))

    # Loss Plot
    plt.subplot(1, 2, 1)
    plt.plot(history.history["loss"], label="Training Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.title("Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    # Accuracy Plot
    plt.subplot(1, 2, 2)
    plt.plot(history.history["accuracy"], label="Training Accuracy")
    plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
    plt.title("Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.tight_layout()
    plt.show()


def calculate_class_weights(labels, num_classes):
    """
    클래스 가중치를 계산합니다.

    Args:
        labels (numpy.ndarray): 레이블 데이터 (원-핫 인코딩).
        num_classes (int): 클래스 개수.

    Returns:
        class_weights (dict): 클래스 가중치 딕셔너리.
    """
    flat_labels = np.argmax(labels, axis=-1).ravel()
    if len(flat_labels) == 0:
        print("No labels found. Returning uniform weights.")
        return {i: 1.0 for i in range(num_classes)}

    flat_labels = flat_labels[flat_labels != -1]  # 패딩된 레이블 제거
    class_weights = compute_class_weight(class_weight="balanced", classes=np.arange(num_classes), y=flat_labels)
    return {i: weight for i, weight in enumerate(class_weights)}


def visualize_segmentation_with_legend(points, true_labels, predicted_labels, num_sample_points, labels, colors):
    """
    클래스 레이블과 색상에 대한 범례를 포함한 세그먼테이션 결과 시각화.

    Args:
        points (numpy.ndarray): (N, 3) 포인트 좌표 (x, y, z).
        true_labels (numpy.ndarray): (N,) 정답 레이블.
        predicted_labels (numpy.ndarray): (N,) 예측 레이블.
        num_sample_points (int): 시각화할 샘플 포인트 수.
        labels (list): 클래스 레이블 이름 리스트.
        colors (list): 클래스별 RGB 색상 리스트.
    """
    if len(points) > num_sample_points:
        indices = np.random.choice(len(points), num_sample_points, replace=False)
        points = points[indices]
        true_labels = true_labels[indices]
        predicted_labels = predicted_labels[indices]

    true_colors = np.array([colors[label] for label in true_labels]) / 255.0
    predicted_colors = np.array([colors[label] for label in predicted_labels]) / 255.0

    fig = plt.figure(figsize=(14, 7))

    ax1 = fig.add_subplot(121, projection='3d')
    ax1.scatter(points[:, 0], points[:, 1], points[:, 2], c=true_colors, s=5)
    ax1.set_title("Ground Truth")
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    ax1.set_zlabel("Z")
    ax1.view_init(30, 30)

    ax2 = fig.add_subplot(122, projection='3d')
    ax2.scatter(points[:, 0], points[:, 1], points[:, 2], c=predicted_colors, s=5)
    ax2.set_title("Predicted Segmentation")
    ax2.set_xlabel("X")
    ax2.set_ylabel("Y")
    ax2.set_zlabel("Z")
    ax2.view_init(30, 30)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    train_and_evaluate()
