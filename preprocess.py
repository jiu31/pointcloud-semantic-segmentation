import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.utils import resample
from joblib import Parallel, delayed
from tensorflow.keras.utils import to_categorical

# 데이터 로드
def load_data(points_file, labels_file):
    """
    포인트 클라우드와 레이블 데이터를 로드합니다.
    
    Args:
        points_file (str): 포인트 클라우드 파일 경로
        labels_file (str): 레이블 파일 경로
    
    Returns:
        point_cloud (numpy.ndarray): 로드된 포인트 클라우드 데이터
        labels (numpy.ndarray): 로드된 레이블 데이터
    """
    print("Loading data...")
    point_cloud = np.loadtxt(points_file)  # (N, 7)
    labels = np.loadtxt(labels_file, dtype=int)  # (N,)
    print(f"Point cloud shape: {point_cloud.shape}")
    print(f"Labels shape: {labels.shape}")
    return point_cloud, labels

# Unlabeled 데이터 제거
def remove_unlabeled(points, labels):
    """
    Unlabeled 데이터를 제거하고 클래스 레이블을 재정의합니다.
    
    Args:
        points (numpy.ndarray): 포인트 클라우드 데이터
        labels (numpy.ndarray): 레이블 데이터
    
    Returns:
        removed_points (numpy.ndarray): Unlabeled가 제거된 포인트 데이터
        removed_labels (numpy.ndarray): Unlabeled가 제거된 레이블 데이터
    """
    mask = labels != 0
    removed_points = points[mask]
    removed_labels = labels[mask]
    removed_labels -= 1  # 클래스 인덱스를 0부터 시작하도록 조정
    return removed_points, removed_labels

# 포인트 클라우드 공간 분할
def partition_point_cloud(point_cloud, labels, grid_size, num_sample_points):
    """
    포인트 클라우드를 그리드 크기 기준으로 공간 분할합니다.
    
    Args:
        point_cloud (numpy.ndarray): 포인트 클라우드 데이터
        labels (numpy.ndarray): 레이블 데이터
    
    Returns:
        filtered_points (list): 그리드별로 필터링된 포인트 데이터
        filtered_labels (list): 그리드별로 필터링된 레이블 데이터
    """
    x_min, x_max = np.min(point_cloud[:, 0]), np.max(point_cloud[:, 0])
    y_min, y_max = np.min(point_cloud[:, 1]), np.max(point_cloud[:, 1])
    x_bins = np.arange(x_min, x_max, grid_size)
    y_bins = np.arange(y_min, y_max, grid_size)

    def process_grid(x_start, y_start):
        x_end = x_start + grid_size
        y_end = y_start + grid_size
        mask = (
            (point_cloud[:, 0] >= x_start) & (point_cloud[:, 0] < x_end) &
            (point_cloud[:, 1] >= y_start) & (point_cloud[:, 1] < y_end)
        )
        grid_points = point_cloud[mask]
        grid_labels = labels[mask]
        if len(grid_points) >= num_sample_points:
            return grid_points, grid_labels
        return None

    results = Parallel(n_jobs=-1)(
        delayed(process_grid)(x_start, y_start) for x_start in x_bins for y_start in y_bins
    )

    filtered_points = []
    filtered_labels = []
    for result in results:
        if result is not None:
            filtered_points.append(result[0])
            filtered_labels.append(result[1])
    return filtered_points, filtered_labels

# 데이터 전처리
def preprocess(partitioned_points, partitioned_labels, num_sample_points, num_classes):
    """
    포인트 클라우드 데이터를 샘플링하고 정규화하며 원-핫 인코딩합니다.
    
    Args:
        partitioned_points (list): 그리드별 포인트 데이터
        partitioned_labels (list): 그리드별 레이블 데이터
    
    Returns:
        processed_point_clouds (numpy.ndarray): 전처리된 포인트 클라우드 데이터
        processed_labels (numpy.ndarray): 전처리된 레이블 데이터
    """
    processed_point_clouds = []
    processed_labels = []
    for grid_points, grid_labels in zip(partitioned_points, partitioned_labels):
        if len(grid_points) >= num_sample_points:
            sampled_indices = np.random.choice(grid_points.shape[0], num_sample_points, replace=False)
            sampled_point_cloud = grid_points[sampled_indices]
            sampled_labels = grid_labels[sampled_indices]
        else:
            padding = num_sample_points - len(grid_points)
            sampled_point_cloud = np.vstack([grid_points, np.zeros((padding, grid_points.shape[1]))])
            sampled_labels = np.hstack([grid_labels, np.full(padding, -1)])
        xyz = sampled_point_cloud[:, :3]
        xyz -= np.mean(xyz, axis=0)
        xyz /= (np.max(np.linalg.norm(xyz, axis=1)) + 1e-8)
        one_hot_labels = to_categorical(sampled_labels, num_classes=num_classes)
        processed_point_clouds.append(xyz)
        processed_labels.append(one_hot_labels)
    return np.array(processed_point_clouds), np.array(processed_labels)

# 데이터 오버샘플링
def oversample_data(points, labels, num_classes):
    """
    데이터셋의 불균형을 완화하기 위해 오버샘플링합니다.

    Args:
        points (numpy.ndarray): 포인트 데이터 (N, 3).
        labels (numpy.ndarray): 레이블 데이터 (N,).
        num_classes (int): 클래스 개수.

    Returns:
        oversampled_points (numpy.ndarray): 오버샘플링된 포인트 데이터.
        oversampled_labels (numpy.ndarray): 오버샘플링된 레이블 데이터.
    """
    oversampled_points = []
    oversampled_labels = []
    for cls in range(num_classes):
        cls_indices = np.where(labels == cls)[0]
        cls_points = points[cls_indices]
        cls_labels = labels[cls_indices]
        if len(cls_indices) > 0:
            resampled_points, resampled_labels = resample(
                cls_points, cls_labels,
                replace=True,
                n_samples=max(len(labels) // num_classes, len(cls_indices)),
                random_state=42
            )
            oversampled_points.append(resampled_points)
            oversampled_labels.append(resampled_labels)
    return np.vstack(oversampled_points), np.hstack(oversampled_labels)

def load_and_preprocess(points_file, labels_file, grid_size, num_sample_points, num_classes):
    """
    데이터 로드부터 전처리, 오버샘플링까지 주요 실행 단계를 수행하고,
    최종 포인트와 레이블 데이터를 출력합니다.
    """
    point_cloud, labels = load_data(points_file, labels_file)
    removed_points, removed_labels = remove_unlabeled(point_cloud, labels)
    partitioned_points, partitioned_labels = partition_point_cloud(removed_points, removed_labels, grid_size, num_sample_points)
    point_clouds, point_cloud_labels = preprocess(partitioned_points, partitioned_labels, num_sample_points, num_classes)
    x_train, x_test, y_train, y_test = train_test_split(
        point_clouds, point_cloud_labels, test_size=0.2, random_state=42
    )
    oversampled_train_points, oversampled_train_labels = oversample_data(x_train, y_train, num_classes)
    return oversampled_train_points, oversampled_train_labels, x_test, y_test
