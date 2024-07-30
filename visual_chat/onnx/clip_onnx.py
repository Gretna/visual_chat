"""
file name:
    clip_onnx.py
author:
    whj
email:
    1020119164@qq.com
summary:
    this file is used to export original clip model to ONNX model.
    first we seperate Clip model to two models: clip vision and clip text.
    And then export these two models to ONNX models correspondingly.
"""

import torch


class ClipTextEncoder(torch.nn.Module):
    """
    Clip text encoder model, used to export to onnx model
    """

    def __init__(self, model):
        """
        model: Clip model
        """
        super().__init__()
        self._model = model

    def forward(self, text):
        """
        Call Clip model's encode_text function to get text feature
        """
        output = self._model.encode_text(text)
        return output / output.norm(p=2, dim=-1, keepdim=True)


class ClipVisionEncoder(torch.nn.Module):
    """
    Clip vision encoder model, used to export to onnx model
    """

    def __init__(self, model) -> None:
        """
        model: Clip model
        """
        super().__init__()
        self._model = model

    def forward(self, x):
        """
        just call Clip model's encode_image function to get image feature
        """
        output = self._model.encode_image(x)
        return output / output.norm(p=2, dim=-1, keepdim=True)


def export_onnx(
    vision_onnx: str,
    text_onnx: str,
    model_name: str = "RN101",
    cache_dir: str = "models",
):
    """
    export clip vision/text model to onnx
    vision_onnx: exported onnx file name for clip vision model
    text_onnx: exported onnx file name for clip text model
    mode_name: which clip model will be exported, see clip.available_models() for details
    cache_dir: download path for clip model weight
    """
    import clip

    model, _ = clip.load(model_name, download_root=cache_dir)
    image = torch.randn(1, 3, 224, 224)  # dummy image input
    text = clip.tokenize(["dummpy input"])

    clipVision = ClipVisionEncoder(model=model)
    clipVision.eval()
    torch.onnx.export(
        clipVision,
        {"x": image},
        vision_onnx,
        dynamic_axes={"x": {0: "batch"}},
        export_params=True,
        input_names=["x"],
        output_names=["vision_feature"],
    )
    clipText = ClipTextEncoder(model=model)
    clipText.eval()
    torch.onnx.export(
        clipText,
        {"text": text},
        text_onnx,
        dynamic_axes={"text": {0: "batch"}},
        export_params=True,
        input_names=["text"],
        output_names=["text_feature"],
    )


if __name__ == "__main__":
    vision_onnx = "clipVision.onnx"
    text_onnx = "clipText.onnx"
    export_onnx(vision_onnx=vision_onnx, text_onnx=text_onnx)
