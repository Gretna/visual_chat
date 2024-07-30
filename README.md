# Visual Chat

This is the visual chat official repository. We focus on implementing an easy way to search images using text.

## Try it out

### Prerequisite

<li>Vitis AI</li>
this project will use AMD vitis AI library to accelerate DNN model inference, so make sure you have the proper hardware. Then follow this instruction to install AMD NPU driver and other libraries <a href="https://ryzenai.docs.amd.com/en/latest/inst.html#">Install Vitis</a>.
Note: <font color='red'>Before you start, you should enable NPU in bios setting, then you can using NPU to accelerate model inference.</font>

<li>Python environment</li>

we suggest using a virtual python environment, like <a href="https://www.anaconda.com/download/">Anaconda</a>. Then you clone this repository and install python dependency packages.

```bash
git clone https://github.com/Gretna/visual_chat.git
cd visual_chat
conda install --yes --file requirements.txt
```

Then run demo.py to try it out.

```bash
python demo.py
```

## Methodology

We modified the origional <a href="https://github.com/openai/CLIP.git">Clip</a> model to meet our requirements with these two aspects:
<li>Accelerate inference using AMD NPU</li>
<li>Using <a href="https://github.com/facebookresearch/faiss.git">Faiss</a> to do quick retrive epspecially for large mount of data</li>
<br>
Clip model was trained in a variety of (image, text) pairs. It can be instructed in natural language to predict the most relevant text snippet. that means if a text is related to an image, then the text vector(encoded by Clip's text encoder) and the vision vector(encoded by Clip's vision encoder) will be very close, this is the basic principle of our method.
We first split origional Clip model into two models: clip vision and clip text which were used to do image/text feature extracion. Then we export these two pytorch models to ONNX models with full precision version. At last, we use vitis AI ONNX quantization tool get the quantization model specific for model accelerate inference with AMD NPU.

### Backbone model select

Clip was trained using various vision model, like resetnet50, resnet101, vit. As vitis AI is optimized for CNN model inference, so we choose resnet as the vision feature extractor. As for the text encoder, we use the origional model.

### Retrieve acceleration

if we have large mount of images, then searching for a specific image will be much time consuming. As we will compare the text feature vector(text model encoded) with all the images' feature vectors(vision model encoded), to find the most similar vectors to this text feature vector. So we use Faiss to do quick searching. For more details, pls refer <a href="https://github.com/Gretna/visual_chat/blob/main/visual_chat/visual_chat.py">Faiss Index</a>.

## Future work
<li>In the future, we will seek for a more powerful vision model to improve performance.</li>
<li>Seek for a stronger Text encoder, fine-tune it for this task</li>
<li>Add support for video content searching</li>
<li>Multi-language support</li>

## Known Issues

<li>when did the quantization using vitis AI library, We found if calibrate_method set to 'vai_q_onnx.PowerOfTwoMethod.MinMSE', the quantization process will consume huge RAM, when running in my machine with 32GB RAM, with calibrate data size 200, the memory was exhausted.</li>

## Contact

if you have any questions, pls feel free to reach 1020119164@qq.com
