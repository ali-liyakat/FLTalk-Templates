"""
Loads MNIST data.
"""



import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Optional torchvision MNIST
try:
    from torchvision import datasets
    _TORCHVISION_OK = True
except Exception:
    _TORCHVISION_OK = False


def load_csv_xy(
    csv_path: str,
    test_size: float = 0.2,
    random_state: int = 42,
    label_col: str | None = None,
):
    """
    CSV classification loader.
    Assumes label is last column if label_col is None.
    Returns: X_train, X_test, y_train, y_test (numpy)
    """
    df = pd.read_csv(csv_path)

    if label_col is None:
        label_col = df.columns[-1]

    X = df.drop(columns=[label_col]).to_numpy().astype(np.float32)
    y = df[label_col].to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    return X_train, X_test, y_train, y_test


def load_mnist(
    data_dir: str = "./data",
    test_size: float = 0.2,
    random_state: int = 42,
    normalize: bool = True,
):
    """
    Loads MNIST (downloads if needed) and returns a train/test split **from MNIST train set**.
    This keeps client-side splitting consistent with your current flow.
    Returns:
      X_train, X_test, y_train, y_test
    Shapes:
      X_* : (N, 28, 28) float32 in [0,1] if normalize=True
      y_* : (N,) int64
    """
    if not _TORCHVISION_OK:
        raise RuntimeError("torchvision not available. Install: pip install torchvision")

    os.makedirs(data_dir, exist_ok=True)

    ds = datasets.MNIST(root=data_dir, train=True, download=True)
    X = ds.data.numpy().astype(np.float32)      # (60000, 28, 28) uint8 -> float
    y = ds.targets.numpy().astype(np.int64)

    if normalize:
        X /= 255.0

    # Client-side split (same as your CSV logic)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    return X_train, X_test, y_train, y_test


def mnist_to_mlp(X):
    """(N,28,28) -> (N,784)"""
    X = np.array(X, dtype=np.float32)
    return X.reshape(X.shape[0], -1)


def mnist_to_cnn(X):
    """
    Keep (N,28,28). Your cnn_mnist.py already accepts (N,28,28) or (N,1,28,28) or (N,784).
    """
    return np.array(X, dtype=np.float32)






import numpy as np

def load_cifar10_resnet(data_dir="./data", resize_224=True, limit_train=None, limit_test=None):
    """
    CIFAR-10 loader for ResNet-18.

    Returns:
      X_train: (N, 3, H, W) float32 in [0,1]
      y_train: (N,) int64
      X_test : (M, 3, H, W) float32 in [0,1]
      y_test : (M,) int64
    """
    from torchvision import datasets, transforms

    tfms = []
    if resize_224:
        tfms.append(transforms.Resize((224, 224)))
    tfms.append(transforms.ToTensor())  # -> (3,H,W), float in [0,1]
    transform = transforms.Compose(tfms)

    train_ds = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform)
    test_ds  = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform)

    def to_numpy(ds, limit=None):
        X, y = [], []
        n = len(ds) if limit is None else min(limit, len(ds))
        for i in range(n):
            img, label = ds[i]
            X.append(img.numpy())     # (3,H,W)
            y.append(label)
        return np.stack(X).astype(np.float32), np.array(y, dtype=np.int64)

    X_train, y_train = to_numpy(train_ds, limit_train)
    X_test, y_test = to_numpy(test_ds, limit_test)

    return X_train, y_train, X_test, y_test
