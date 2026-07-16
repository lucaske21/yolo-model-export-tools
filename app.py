import argparse
from pathlib import Path

from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description="Export a YOLO .pt model to ONNX.")
    parser.add_argument("--model", default="yolo11n.pt", help="Path to the .pt model file.")
    parser.add_argument("--imgsz", type=int, default=640, help="Input image size.")
    # adding batch size
    parser.add_argument("--batch", type=int, default=1, help="ONNX set model batch size")
    parser.add_argument("--opset", type=int, default=12, help="ONNX opset version.")
    parser.add_argument("--half", action="store_true", help="Export in FP16 (GPU only).")
    parser.add_argument("--dynamic", action="store_true", help="Enable dynamic input shapes.")
    parser.add_argument("--no-simplify", action="store_true", help="Disable graph simplification.")
    return parser.parse_args()


def main():
    args = parse_args()
    model = YOLO(args.model)
    # setup the output name with basename-b{batch}
    path = model.export(
        format="onnx",
        imgsz=args.imgsz,
        opset=args.opset,
        half=args.half,
        batch=args.batch,
        dynamic=args.dynamic,
        simplify=not args.no_simplify,
    )
    exported = Path(path)
    output = exported.with_name(f"{exported.stem}-b{args.batch}{exported.suffix}")
    exported.replace(output)
    print(f"Exported ONNX model to: {output}")


if __name__ == "__main__":
    main()
