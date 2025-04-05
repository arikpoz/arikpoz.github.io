
---
title: "Fast Image Loading with NVIDIA nvImageCodec"
date: 2025-04-04
draft: true
---

In deep learning pipelines, especially those involving image data, **data loading and preprocessing** often become major bottlenecks. Traditionally, image decoding is performed using libraries like [OpenCV](https://docs.opencv.org/4.x/index.html) or [Pillow](https://pillow.readthedocs.io/en/stable/), which rely on CPU-based processing. After decoding, the data must be transferred to GPU memory for further operations. But what if the decoding process itself could be performed directly on the GPU? Could this lead to faster performance?

Enter **`nvImageCodec`** ([documentation](https://docs.nvidia.com/cuda/nvimagecodec/), [GitHub](https://github.com/NVIDIA/nvImageCodec))—NVIDIA's GPU-accelerated image decoding library designed for high-throughput pipelines.

---

## 🔍 What is `nvImageCodec`?

`nvImageCodec` is a high-performance image codec optimized for GPU acceleration. It is designed for scenarios like model training and batch inference, where decoding thousands of images quickly is critical. The library supports decoding (bytes to pixels) and encoding (pixels to bytes) for various common image formats. However, not all formats are fully supported on the GPU. Some, like PNG and WebP, fall back to CPU-based decoding. Below is a summary of supported formats:

### ✅ Format Support:

| Format    | GPU Decode | GPU Encode | Notes                         |
| --------- | ---------- | ---------- | ----------------------------- |
| JPEG      | ✅ Yes      | ✅ Yes      | Fastest, hardware-accelerated |
| JPEG 2000 | ✅ Yes      | ✅ Yes      |                               |
| TIFF      | ✅ Yes      | ❌ No (planned) | CUDA decoder                  |
| PNG       | ❌ No (planned) | ❌ No (planned) | CPU only                      |
| WebP      | ❌ No       | ❌ No       | CPU only                      |

---

## 🌟 What Was Benchmarked?

We compared the performance of:

- **OpenCV**: CPU-based decoding followed by PIL transformations.
- **`nvImageCodec`**: GPU-based decoding with tensor transformations.

### Benchmark Details:

- **Dataset**: 1000 JPEG images from the [ImageNet Sample Images dataset](https://github.com/EliSchwartz/imagenet-sample-images) (credit: [Eli Schwartz](https://github.com/EliSchwartz)).
- **Model**: ResNet18 for inference.
- **Transform Pipeline**: Resize and crop applied to all images.

Each benchmark was run **10 times** (plus 1 warmup iteration), and the **average times** were recorded for:

- 🧪 **Loading**: Decoding, resizing, and tensor conversion.
- ⚡ **Inference**: Model forward pass.
- ⏱️ **Total**: Combined loading and inference time.

All benchmarks were conducted in [**Google Colab**](https://colab.research.google.com/) using a T4 GPU instance.

[**Run this code in Google Colab**](https://colab.research.google.com/drive/1UUqyHYMaiv3evWaHbuW-OKhR91ferLVl?usp=sharing) to try it yourself.


---

## 🛠️ Setup in Colab

### Install Dependencies and Load Dataset

```python
!pip install nvidia-nvimgcodec-cu11 opencv-python-headless
!git clone https://github.com/EliSchwartz/imagenet-sample-images.git
```

### Prepare the Images

```python
import os, shutil
from pathlib import Path

source_dir = Path("imagenet-sample-images")
dest_dir = Path("benchmark_images")
dest_dir.mkdir(exist_ok=True)

all_images = list(source_dir.glob("*.JPEG"))
for img in all_images:
    shutil.copy(img, dest_dir / img.name)

image_paths = sorted(list(dest_dir.glob("*.JPEG")))
print(f"Prepared {len(image_paths)} images.")
```

### Define Model and Preprocessing

```python
import torch
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.transforms import Resize, CenterCrop

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet18(pretrained=True).to(device).eval()

transform = transforms.Compose([
    Resize(256),
    CenterCrop(224),
])
```

---

## 🧲 Benchmark Functions (10x Repeated Runs)

### OpenCV Benchmark

```python
def run_opencv_inference(image_paths, runs=10):
    import time, numpy as np
    from PIL import Image

    load_times, infer_times = [], []
    for run_idx in range(runs + 1):
        imgs = []
        t0 = time.time()
        for path in image_paths:
            img = cv2.imread(str(path))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            img = transform(img)
            img = transforms.ToTensor()(img)
            imgs.append(img)
        batch = torch.stack(imgs).to(device)
        load_time = time.time() - t0

        t1 = time.time()
        with torch.no_grad():
            model(batch)
        infer_time = time.time() - t1

        if run_idx == 0:
            print(f"Run {run_idx + 1}: Warmup iteration (not included in mean). Loading Time = {load_time:.4f}s, Inference Time = {infer_time:.4f}s")
        else:
            load_times.append(load_time)
            infer_times.append(infer_time)
            print(f"Run {run_idx + 1}: Loading Time = {load_time:.4f}s, Inference Time = {infer_time:.4f}s")

    return np.mean(load_times), np.mean(infer_times)

opencv_load, opencv_infer = run_opencv_inference(image_paths)
```

### nvImageCodec Benchmark

```python
def run_nvimagecodec_inference(image_paths, runs=10):
    import time, numpy as np
    decoder = nvimgcodec.Decoder(device_id=0)

    load_times, infer_times = [], []
    for run_idx in range(runs + 1):
        imgs = []
        t0 = time.time()
        for path in image_paths:
            with open(path, 'rb') as f:
                data = f.read()
            nv_img = decoder.decode(data)
            img = torch.as_tensor(nv_img.cuda()).permute(2, 0, 1).float().div(255)
            img = transform(img)
            imgs.append(img)
        batch = torch.stack(imgs).to(device)
        load_time = time.time() - t0

        t1 = time.time()
        with torch.no_grad():
            model(batch)
        infer_time = time.time() - t1

        if run_idx == 0:
            print(f"Run {run_idx + 1}: Warmup iteration (not included in mean). Loading Time = {load_time:.4f}s, Inference Time = {infer_time:.4f}s")
        else:
            load_times.append(load_time)
            infer_times.append(infer_time)
            print(f"Run {run_idx + 1}: Loading Time = {load_time:.4f}s, Inference Time = {infer_time:.4f}s")

    return np.mean(load_times), np.mean(infer_times)

nv_load, nv_infer = run_nvimagecodec_inference(image_paths)
```

---

## 📊 Results & Visualization

```python
import pandas as pd
import matplotlib.pyplot as plt

results = pd.DataFrame({
    "Method": ["OpenCV", "nvImageCodec"],
    "Loading Time (s)": [opencv_load, nv_load],
    "Inference Time (s)": [opencv_infer, nv_infer],
    "Total Time (s)": [
        opencv_load + opencv_infer,
        nv_load + nv_infer
    ],
})

print(results)

results.plot(x="Method", y=["Loading Time (s)", "Inference Time (s)", "Total Time (s)"],
             kind="bar", figsize=(10, 6))
plt.title("OpenCV vs. nvImageCodec on 1000 ImageNet JPEGs (10-run average)")
plt.ylabel("Seconds")
plt.grid(True)
plt.show()
```

---

## ✅ Summary

| Method           | Loading Time (s) | Inference Time (s) | Total Time (s) |
| ---------------- | ---------------- | ------------------ | -------------- |
| **OpenCV**       | 6.08343          | 0.00349            | 6.08693        |
| **nvImageCodec** | 2.78262          | 0.00323            | 2.78585        |

By leveraging the T4 GPU, `nvImageCodec` achieves a **2.18x speedup** in JPEG loading times by performing decoding directly on the GPU. This eliminates CPU bottlenecks and enables a more efficient data pipeline.

For workflows heavily reliant on JPEGs, integrating `nvImageCodec` into your training or inference pipeline can deliver substantial performance improvements with minimal effort.

**Tip**: Before integrating, ensure that loading time is indeed a bottleneck in your pipeline. For example, test by preloading a single image or skipping loading altogether to simulate random data. In training pipelines, prefetching images in parallel with GPU processing is also a common optimization strategy.
