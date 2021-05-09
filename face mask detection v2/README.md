
**There are 3 different classes annotated**:
- **No mask**: No mask at all.
- **Mask worn incorrectly**: Partially covered face. It detects whether people have wore a mask properly or not.
 - **Mask**: Mask covers the essential parts.

![](https://github.com/shejz/face-mask-detector/blob/main/face%20mask%20detection%20v2/mask_wear.gif)

[Dataset](https://www.sciencedirect.com/science/article/pii/S2352648320300362?via%3Dihub)

MaskedFace-Net – A dataset of correctly/incorrectly masked face images in the context of COVID-19

**Data preparation**

``` 
├── dataset
│   ├── Useless_Mask 
│   │   ├── 00001.png
│   │   ├── 00002.png
│   │   ├── ...
│   ├── Mask
│   │   ├── 00001.png
│   │   ├── 00002.png
│   │   ├── ...
│   ├── No_Mask
│   │   ├── 00001.png
│   │   ├── 00002.png
│   │   ├── ...
└───└─── 
```
