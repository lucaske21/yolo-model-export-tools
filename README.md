# Tools for yolo series model exporting

## Setup env via uv

[uv](https://docs.astral.sh/uv/) is a fast Python package and project manager. Install it first if you don't have it.

### 1. Install uv

Windows (PowerShell):
```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

macOS / Linux:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Verify the install:
```bash
uv --version
```

### 2. Create a virtual environment

Create a `.venv` in the project folder pinned to a specific Python version:
```bash
uv venv --python 3.10
```

Activate it:

- Windows (PowerShell):
  ```powershell
  .venv\Scripts\Activate.ps1
  ```
- macOS / Linux:
  ```bash
  source .venv/bin/activate
  ```

### 3. Install dependencies

Install Ultralytics (which pulls in PyTorch) and the ONNX exporter dependencies:
```bash
uv pip install ultralytics onnx onnxruntime onnxslim
```

> Tip: for GPU export, install the CUDA build of PyTorch that matches your driver, e.g.:
> ```bash
> uv pip install torch --index-url https://download.pytorch.org/whl/cu121
> ```

Confirm the install:
```bash
uv pip list
```

## Export a `.pt` model to ONNX with Ultralytics

### Option A — CLI (one-liner)

```bash
yolo export model=yolo11n.pt format=onnx
```

This downloads `yolo11n.pt` (if not present) and writes `yolo11n.onnx` next to it.

Common export options:
```bash
yolo export model=yolo11n.pt format=onnx imgsz=640 opset=12 half=False dynamic=False simplify=True
```

| Option     | Meaning                                  |
| ---------- | ---------------------------------------- |
| `imgsz`    | Input resolution (single int or `h,w`)   |
| `opset`    | ONNX opset version                       |
| `half`     | Export in FP16 (GPU only)                |
| `dynamic`  | Dynamic batch / input shapes             |
| `simplify` | Run `onnxslim` to simplify the graph     |
| `batch`    | Fixed batch size                         |

### Option B — Python script

Use the provided `app.py`:
```bash
python app.py --model yolo11n.pt --imgsz 640 --opset 12
```

Or inline:
```python
from ultralytics import YOLO

model = YOLO("yolo11n.pt")
model.export(format="onnx", imgsz=640, opset=12, simplify=True)
```

### Verify the exported model

```python
import onnx

onnx_model = onnx.load("yolo11n.onnx")
onnx.checker.check_model(onnx_model)
print("ONNX model is valid.")
```

Quick inference sanity check with onnxruntime:
```python
import numpy as np
import onnxruntime as ort

sess = ort.InferenceSession("yolo11n.onnx", providers=["CPUExecutionProvider"])
inp = sess.get_inputs()[0]
dummy = np.random.rand(*[d if isinstance(d, int) else 1 for d in inp.shape]).astype(np.float32)
outputs = sess.run(None, {inp.name: dummy})
print("Output shapes:", [o.shape for o in outputs])
```