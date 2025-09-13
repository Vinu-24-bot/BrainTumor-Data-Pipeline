from typing import List
from google.cloud import storage
import numpy as np
import cv2


def upload_dir_to_gcs(local_dir: str, bucket_name: str, prefix: str = "raw") -> List[str]:
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    uploaded = []
    from pathlib import Path

    for p in Path(local_dir).glob("**/*"):
        if p.is_file():
            blob = bucket.blob(f"{prefix.rstrip('/')}/{p.name}")
            blob.upload_from_filename(str(p))
            uploaded.append(blob.name)
    return uploaded


def list_images(bucket_name: str, prefix: str, limit: int = 50) -> List[str]:
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blobs = bucket.list_blobs(prefix=prefix.rstrip("/"))
    names = [b.name for b in blobs if b.name.lower().endswith(".png")]
    return names[:limit]


def download_image_np(bucket_name: str, blob_name: str) -> np.ndarray:
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    data = blob.download_as_bytes()
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
    return img
