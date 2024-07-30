"""
file name:
    visual_chat.py
author:
    whj
email:
    1020119164@qq.com
summary:
    this file is the visual_chat class file.
"""

import onnxruntime
from datasets import Dataset
from PIL import Image
import numpy as np
import os
from .util.transform import transform
import faiss


class VisualChat:
    def __init__(
        self,
        folder: str,
        vision_onnx: str,
        text_onnx: str,
        faiss_index: str = "img_index.faiss",
        vaip_config: str = "vaip_config.json",
        n_px: int = 224,
    ) -> None:
        """
        folder: searching folder that contains images
        vision_onnx: clip vision onnx model file
        text_onnx: clip text onnx model file
        faiss_index: faiss indexing file name saved to disk
        vaip_config: vitis runtime config file
        n_px: image pixel input to model
        """
        self._folder = folder
        self._faiss_index = faiss_index
        self._img_preprocess = transform(n_px=n_px)
        self._init_vision_model(vision_onnx, vaip_config)
        self._init_text_model(text_onnx, vaip_config)
        self._create_faiss_index()

    def _init_vision_model(self, vision_onnx: str, vaip_config: str):
        """
        vision_onnx: quantized clip vision encoder ONNX model file
        vaip_config: vitis ai config file
        """
        self._vision_session = onnxruntime.InferenceSession(
            vision_onnx,
            providers=["VitisAIExecutionProvider"],
            provider_options=[{"config_file": vaip_config}],
        )

    def _init_text_model(self, text_onnx: str, vaip_config: str):
        """
        text_onnx: quantized clip text encoder ONNX model file
        vaip_config: vitis ai config file
        """
        self._text_session = onnxruntime.InferenceSession(
            text_onnx,
            providers=["VitisAIExecutionProvider"],
            provider_options=[{"config_file": vaip_config}],
        )

    def _create_dataset(self):
        """
        generate dataset for images
        """
        images = []
        files = []
        for file in os.listdir(self._folder):
            file_path = os.path.join(self._folder, file)
            try:
                image = Image.open(file_path)
                images.append(image)
                files.append(file_path)
            except Exception as e:
                print(f"error while loading:{file_path},error info: {e}")
                continue
        return Dataset.from_dict({"image": images, "source": files})

    def _encode_image(self, image: Image):
        """
        encode image to get image feature
        img: Image object
        """
        img = self._img_preprocess(image).unsqueeze(0)
        vision_out = self._vision_session.run([], {"x": img.numpy()})
        return vision_out[0][0]

    def _encode_text(self, text: str):
        """
        encode text to get text feature 
        text: text to be encoded by clip text encoder
        """
        import clip

        text = clip.tokenize([text])
        text_out = self._text_session.run(
            None,
            {"text": text.numpy()},
        )
        return text_out[0][0]

    def _create_faiss_index(self):
        """
        create faiss index for quick retrieve
        """
        dataset = self._create_dataset()
        img_embeddings = dataset.map(
            lambda data: {"img_embeddings": self._encode_image(data["image"])}
        )
        img_embeddings.add_faiss_index(
            column="img_embeddings", metric_type=faiss.METRIC_INNER_PRODUCT
        )
        img_embeddings.save_faiss_index("img_embeddings", self._faiss_index)

    def search(self, text: str, k: int = 3):
        """
        text: searching string to match images
        k: found k images with top k similarities
        """
        text_feature = self._encode_text(text)
        dataset = self._create_dataset()
        dataset.load_faiss_index("img_embeddings", self._faiss_index)
        scores, imgs = dataset.get_nearest_examples("img_embeddings", text_feature, k=k)
        return scores, imgs["source"]
