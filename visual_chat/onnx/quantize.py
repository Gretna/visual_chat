"""
file name:
    quantize.py
author:
    whj
email:
    1020119164@qq.com
summary:
    this file is used to do the quantization operation for ONNX model running on AMD IPU.
"""

from util.transform import transform
from onnxruntime.quantization.calibrate import CalibrationDataReader
import random, os
from PIL import Image
import vai_q_onnx


def _convert_image_to_rgb(image):
    return image.convert("RGB")


class CalibDataloader(CalibrationDataReader):
    """
    this class is used to get calibration data for clip vision model
    """

    def __init__(self, image_folder: str, img_npx: int = 344, size: int = 100):
        """
        image_folder: folder which contains images
        img_npx: image px size input to the model
        size: calibration data size
        """
        import clip

        super().__init__()
        files = os.listdir(image_folder)
        random.shuffle(files)
        # _, preprocess = clip.load("RN101", download_root="models")
        preprocess = transform(img_npx)
        calibData = []
        for file in files[:size]:
            file_name = os.path.join(image_folder, file)
            img = Image.open(file_name)
            calibData.append(
                {
                    "x": preprocess(img).unsqueeze(0).numpy(),
                }
            )
        self._data = iter(calibData)

    def get_next(self):
        return next(self._data, None)


def quantize_clip_vision(
    model: str, output_name: str = "clipVsion_q.onnx", size: int = 100
):
    """
    model: original clip vision onnx model
    output_name: quantized onnx model
    size: calibration data size used to quantize
    """
    datareader = CalibDataloader(size=size)
    vai_q_onnx.quantize_static(
        model,
        output_name,
        calibration_data_reader=datareader,
        quant_format=vai_q_onnx.QuantFormat.QDQ,
        calibrate_method=vai_q_onnx.PowerOfTwoMethod.MinMSE,
        activation_type=vai_q_onnx.QuantType.QUInt8,
        weight_type=vai_q_onnx.QuantType.QInt8,
        enable_dpu=True,
        extra_options={"ActivationSymmetric": True},
    )