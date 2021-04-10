
# **Face Mask Detection** 😷
Train a custom deep learning model to detect whether a person is or is not wearing a mask.

This dataset consists of 1,376 images belonging to two classes:

- with_mask: 690 images
- without_mask: 686 images

## **How was our face mask dataset created?**
- Create an artificial dataset of people wearing masks.

Techniques used to create images:
1. Taking normal images of faces.
2. Then creating a custom computer vision Python script to add face masks to them, thereby creating an artificial (but still real-world applicable) dataset.

**Facial landmarks** allow us to automatically infer the location of facial structures, including:

- Eyes
- Eyebrows
- Nose
- Mouth
- Jawline

**Steps**:
1. Start with an image of a person not wearing a face mask.
2. Apply face detection to compute the bounding box location of the face in the image
3. Extract the face Region of Interest (ROI)
4. Apply facial landmarks, allowing us to localize the eyes, nose, mouth, etc.
5. Get image of a mask, and align it on top of the face properly. This mask will be automatically applied to the face by using the facial landmarks (namely the points along the chin and nose) to compute where the mask will be placed. The mask is then resized and rotated, placing it on the face
6. Repeat this process for all of our input images.

![](https://i.postimg.cc/d0wRgmpX/face-steps.jpg)

<a id="mask_detection_"></a>
# Face Mask Detection
It detects whether people have wore a mask properly or not.

<img src="https://i.postimg.cc/zBMdg2XM/peace2.jpg" width="400"/>   
<img src="https://i.postimg.cc/43fFkhS8/my-face.jpg" width="400"/>
<img src="https://i.postimg.cc/5tsKx2cf/peace.jpg" width="400"/>/>   
<img src="https://i.postimg.cc/LXhChFRL/peace-mask.jpg" width="400"/>



## **Project structure**

- **train_mask_detector.py**: Accepts our input dataset and fine-tunes **MobileNetV2** upon it to create our mask_detector.model. A training history plot.png containing accuracy/loss curves is also produced
- **detect_mask_image.py**: Performs face mask detection in static images.
- **detect_mask_video.py**: Using your webcam, this script applies face mask detection to every frame in the stream.

```bash
.
├── dataset
│   ├── with_mask [690 entries]
│   └── without_mask [686 entries]
├── examples
│   ├── face1.jpg
│   ├── face2.jpg
│   └── face3.jpg
├── face_detector
│   ├── deploy.prototxt
│   └── res10_300x300_ssd_iter_140000.caffemodel
├── detect_mask_image.py
├── detect_mask_video.py
├── mask_detector.model
├── loss_accuracy_plot.png
└── train_mask_detector.py

5 directories, 10 files

```
