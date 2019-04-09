# Merge BN v1.0.0

This the scripts is used to merge the batch normalize and the convolution of yolov3 or yolov3-tiny.

## Get Started

1.python 

2.numpy

## How to run

### 1. initialization

```
cfg_name = 'PED_DET_001_middle.cfg'   
weight_file = 'PED_DET_001.weights'
out_weight_file = 'PED_DET_001_0.weights'

cfg_name: '.cfg' file path of yolov3 or yolov3-tiny
weight_file: '.weights' file path of yolov3 or yolov3-tiny
out_weight_file: '.weights' file path of output
```
## 2.cfg_file

#### yolov3:

yolov3 训练cfg : PED_DET_001_middle.cfg

yolov3 merge bn cfg:PED_DET_001.cfg

### 3.Output

The output is the '.weights' file.

When you use the weights file in darknet, you should set the all batch_normalize=0 of the '.cfg' file.

Compare the orgin weights file, the output weights file accuracy isn't change, but the speed improve 20%.

**Note:** if you use the darknet AlexeyAB version, you should check the size of your weights file header is equal to 5*sizeof(int). If it is not equal to 5 * sizeof(int), you should change here:

header = np.fromfile(fp, **dtype = np.int32, count = 5**)

Make sure your weight file is equal to the header

