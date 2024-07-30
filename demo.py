"""
file name:
    demo.py
author:
    whj
email:
    1020119164@qq.com
summary:
    visual chat's demo script, include onnx model quantizaion and chat with images
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"


def chat_demo(vision_onnx: str = None, text_onnx: str = None):
    """
    visual chat demo
    vision_onnx: quantized clip vision onnx model
    text_onnx: clip text onnx model
    """
    from visual_chat import VisualChat
    from matplotlib import image as mpimg
    from matplotlib import pyplot as plt

    if not vision_onnx or not text_onnx:
        # download quantized model form huggingface
        from huggingface_hub import snapshot_download

        snapshot_download("agier9/visual_chat", local_dir="onnx_model")
        vision_onnx = "onnx_model/quantizedVision.onnx"
        text_onnx = "onnx_model/clipText.onnx"
    image_folder = "picture/"
    vchat = VisualChat(image_folder, vision_onnx, text_onnx)
    text = "a photo of a dog"
    scores, imgs = vchat.search(text, k=4)
    fig, axs = plt.subplots(nrows=2, ncols=2)
    fig.subplots_adjust(hspace=0.3,wspace=0.2)
    axs = axs.flatten()
    for p,ax,img_file in zip(scores,axs,imgs):
        ax.set_title(str(p))
        image = mpimg.imread(img_file)
        ax.imshow(image)
    plt.show()


def quantize(image_folder: str, size: int = 200):
    """
    quantize clip vision onnx model to fit for vitis AI lib
    image_folder: folder of images used as calibration data to do quantization
    size: calibration data size used to quantize
    """
    from visual_chat.onnx import export_onnx
    from visual_chat.onnx import quantize_clip_vision

    clip_vision_onnx = "clipVision.onnx"
    clip_text_onnx = "clipText.onnx"
    # export clip pytorch model to onnx model
    export_onnx(clip_vision_onnx, clip_text_onnx)

    # do the quantization
    quantize_clip_vision(clip_vision_onnx, "quantizedVision.onnx", image_folder, size)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser("visual chat demo")
    parser.add_argument("--usage", choices=["quantize", "chat"], default="chat")

    args = parser.parse_args()
    usage = args.usage
    if usage == "quantize":
        # do quantization
        # you can use the imageNet-1K dataset for quantization
        image_folder = "your image folder"
        quantize(image_folder=image_folder, size=100)
    else:
        chat_demo()
