"""
Export the transformer model to ONNX for browser inference.

Example:
  python3 training/export_onnx.py --checkpoint models/transformer_v42.pt --output frontend/assets/planet_model.onnx
"""

import argparse
from pathlib import Path
import sys

import torch


def load_model(checkpoint_path: Path):
    root_dir = Path(__file__).resolve().parents[1]
    if str(root_dir) not in sys.path:
        sys.path.insert(0, str(root_dir))
    from backend.app.ml.transformer import TransformerForGeneration

    model = TransformerForGeneration()
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    return model


def export_onnx(model, output_path: Path, opset: int, max_seq_len: int):
    dummy_input = torch.zeros((1, max_seq_len), dtype=torch.long)
    export_kwargs = dict(
        opset_version=opset,
        input_names=["input_ids"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "sequence"},
            "logits": {0: "batch", 1: "sequence"},
        },
    )
    try:
        torch.onnx.export(model, dummy_input, str(output_path), dynamo=False, **export_kwargs)
    except TypeError:
        # Older/alternate torch builds may not support the dynamo kwarg.
        torch.onnx.export(model, dummy_input, str(output_path), **export_kwargs)


def quantize_int8(input_path: Path, output_path: Path):
    from onnxruntime.quantization import QuantType, quantize_dynamic

    quantize_dynamic(
        str(input_path),
        str(output_path),
        weight_type=QuantType.QInt8,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="Path to .pt checkpoint")
    parser.add_argument("--output", required=True, help="Path to output .onnx")
    parser.add_argument("--opset", type=int, default=18)
    parser.add_argument("--max-seq-len", type=int, default=64)
    parser.add_argument("--quantize-int8", action="store_true")
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model = load_model(checkpoint_path)
    export_onnx(model, output_path, args.opset, args.max_seq_len)

    if args.quantize_int8:
        quant_path = output_path.with_name(output_path.stem + "_int8.onnx")
        quantize_int8(output_path, quant_path)

    print(f"Exported ONNX to {output_path}")
    if args.quantize_int8:
        print(f"Exported INT8 to {quant_path}")


if __name__ == "__main__":
    main()
