# ONNX-SAM2-Segment-Anything
![!ONNX-SAM2-Segment-Anything](https://github.com/ibaiGorordo/ONNX-SAM2-Segment-Anything/raw/main/doc/img/sam2_mask_with_boxes.png)

# Important
- Still in development, use it at your own risk.
- For now only the image prediction is available, the video prediction will be available soon.
- Other limitations: Only default resolution (1024x1024)

# Installation
```shell
git clone https://github.com/5PB-3-4/ONNX-SAM2.1-Segment-Anything.git -b sam2.1
cd ONNX-SAM2-Segment-Anything
pip install -r requirements.txt
```
### ONNX Runtime
For Nvidia GPU computers:
`pip install onnxruntime-gpu`

Otherwise:
`pip install onnxruntime`

# ONNX model
- Read [this](https://github.com/ibaiGorordo/ONNX-SAM2-Segment-Anything/issues/18)
- The encoder and decoder models can download [this](https://huggingface.co/rectlabel/segment-anything-onnx-models/tree/main)

# Original Semgent Anything Model 2 (SAM2)
The original SAM2 model can be found in this repository: [SAM2 Repository](https://github.com/facebookresearch/segment-anything-2)
- The License of the models is Apache 2.0: [License](https://github.com/facebookresearch/segment-anything-2/blob/main/LICENSE)

# Examples

![!ONNX-SAM2-Segment-Anything-iMAGE](https://github.com/ibaiGorordo/ONNX-SAM2-Segment-Anything/raw/main/doc/img/sam2_masked_img.jpg)

## **Image inference**:
Runs the image segmentation model on an image given some points defined in the script.
 ```shell
 python demo_sam2.exe -i <your image file path> -e model/sam2.1_small_preprocess.onnx -d model/sam2.1_small.onnx
 ```

Usage:
- `point_coords`: This is a list of 2D numpy arrays, where each element in the list correspond to a different label. For example, for 3 different labels, the list will contain 3 numpy arrays. Each numpy array contains Nx2 points, where N is the number of points and the second axis contains the X,Y coordinates (of the original image)
- `point_labels`: This is a list of 1D numpy arrays, where each element in the list correspond to a different label. For example, for 3 different labels, the list will contain 3 numpy arrays. Each numpy array contains N points, where N is the number of points. The value can be 0 or 1, where 0 represents a negative value and 1 a positive value, i.e. the objects is present at that pixel location.


![!ONNX-SAM2-Segment-Anything](https://github.com/ibaiGorordo/ONNX-SAM2-Segment-Anything/raw/main/doc/img/sam2_annotation.gif)
## **SAM2 Annotation App**:
A minimal GUI to annotate images with the SAM2 model.
 ```shell
python demo_sam2.exe -i <your image file path> -e model/sam2.1_small_preprocess.onnx -d model/sam2.1_small.onnx -a
 ```
Annotation Controls (Video: https://youtu.be/9lW3_g1fjnA?si=X49Vz1ow45NMMYVn)
- **Left click**: Adds a positive point, but if another point is close enough, it will delete it
- **Right click**: Adds a negative point
- **Left click and drag**: Draws a rectangle
- **Add label button**: Adds a new label for annotation
- **Delete label button**: Deletes the last label

# References:
* SAM2 Repository: [https://github.com/facebookresearch/segment-anything-2](https://github.com/facebookresearch/segment-anything-2)
* Original Repository: [https://github.com/ibaiGorordo/ONNX-SAM2-Segment-Anything)

