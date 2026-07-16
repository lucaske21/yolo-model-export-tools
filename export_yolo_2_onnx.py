#!/usr/bin/env python3

import argparse
import os
import onnx
import torch
from onnx.external_data_helper import _get_all_tensors


class _ExportWrapper(torch.nn.Module):
    """Wrap a YOLO seg model so it exposes only the two real outputs.

    The underlying model returns ``((detection, protos), aux_dict)``. The
    ONNX exporter would otherwise leak every tensor in ``aux_dict`` as an
    extra graph output, so we keep just ``output0`` (detection) and
    ``output1`` (mask prototypes).
    """

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        out = self.model(x)
        # Normalize to the (detection, protos) pair regardless of nesting.
        if isinstance(out, (list, tuple)):
            out = out[0]
        detection, protos = out[0], out[1]
        return detection, protos


def export_onnx(
    pt_path,
    onnx_path,
    batch_size=1,
    img_size=640,
    opset=19,
):
    print(f"Loading model: {pt_path}")

    # Load checkpoint. Ultralytics checkpoints pickle custom classes, so
    # weights_only must be False (PyTorch >= 2.6 defaults it to True).
    # Only safe because these are trusted, locally-produced weights.
    ckpt = torch.load(pt_path, map_location="cpu", weights_only=False)

    # Support both checkpoint formats
    if isinstance(ckpt, dict):
        if "model" in ckpt:
            model = ckpt["model"]
        elif "ema" in ckpt:
            model = ckpt["ema"]
        else:
            raise RuntimeError("Cannot find model in checkpoint.")
    else:
        model = ckpt

    model.float()
    model.eval()

    model = _ExportWrapper(model)
    model.eval()

    dummy = torch.randn(
        batch_size,
        3,
        img_size,
        img_size,
        dtype=torch.float32,
    )

    print("Exporting ONNX...")

    torch.onnx.export(
        model,
        dummy,
        onnx_path,
        export_params=True,
        opset_version=opset,
        do_constant_folding=True,
        input_names=["images"],
        output_names=[
            "output0",
            "output1",
        ],
        dynamic_axes={
            "images": {
                0: "batch",
            },
            "output0": {
                0: "batch",
            },
            "output1": {
                0: "batch",
            },
        },
    )

    # Re-save as a single self-contained file (weights inlined, no sidecar
    # ".onnx_data"/".data" file), even for large models.
    onnx_dir = os.path.dirname(os.path.abspath(onnx_path))

    # Load metadata only first so we can see which sidecar files are
    # referenced (loading the data eagerly clears these locations).
    model_proto = onnx.load(onnx_path, load_external_data=False)
    external_files = set()
    for tensor in _get_all_tensors(model_proto):
        for entry in tensor.external_data:
            if entry.key == "location":
                external_files.add(
                    os.path.join(onnx_dir, entry.value)
                )

    # Now pull the weights into memory and write a single inlined file.
    onnx.load_external_data_for_model(model_proto, onnx_dir)
    onnx.save_model(
        model_proto,
        onnx_path,
        save_as_external_data=False,
    )

    # Remove now-orphaned sidecar weight files.
    for path in external_files:
        if os.path.isfile(path) and os.path.abspath(path) != os.path.abspath(onnx_path):
            os.remove(path)

    print(f"ONNX saved to: {onnx_path}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--weights",
        required=True,
        help="YOLO26-seg .pt model",
    )

    parser.add_argument(
        "--output",
        default="yolo26-seg.onnx",
        help="Output ONNX filename",
    )

    parser.add_argument(
        "--batch",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
    )

    parser.add_argument(
        "--opset",
        type=int,
        default=19,
    )

    args = parser.parse_args()

    export_onnx(
        pt_path=args.weights,
        onnx_path=args.output,
        batch_size=args.batch,
        img_size=args.imgsz,
        opset=args.opset,
    )


if __name__ == "__main__":
    main()