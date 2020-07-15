# Improving Keypoint Matching Using a Landmark-Based Image Representation

Motivated by the need to improve the performance of visual loop closure verification via multi-view geometry (MVG) under significant illumination and viewpoint changes, we propose a keypoint matching method that uses landmarks as an intermediate image representation in order to leverage the power of deep learning. In environments with various changes, the traditional verification method via MVG may encounter difficulty because of their inability to generate a sufficient number of correctly matched keypoints. Our method exploits the excellent invariance properties of convolutional neural network (ConvNet) features, which have shown outstanding performance for matching landmarks between images. By generating and matching landmarks first in the images and then matching the keypoints within the matched landmark pairs, we
can significantly improve the quality of matched keypoints in terms of precision and recall measures. 

The source code is [hosted on GitHub](https://github.com/Hansry/Keypoint-Matching-Based-on-Landmark-Representation).

## Building and Running

## Dependencies:
&emsp; &emsp;Python: 3.5.6  
&emsp; &emsp;Pytorch: 0.4.1  
&emsp; &emsp;Torchvision: 0.1.8  
&emsp; &emsp;OpenCV: 3.10  

### Building 

This project is built in Python and requires no training, so you can directly run the code.

### Demo Sequence
  When the environment is configured, try processing the demo sequence: [here is the dataset used in paper](https://pan.baidu.com/s/1ohZmhpvq-6ivH40kbOx7AQ) and the password is `e644`.

  1. Extract the dataset to a directory, then generate the potential loop clousre dataset:
        ```bash
        python potential_loop_closure.py -dt UAcampus -in 647 -dth 0.7
        ```
     Using the `python potential_loop_closure.py -h` to check the meaning of the variables, note that the `AY`,`BY` means the true positive in query and train image respectively, the `AN`, `BN` means the false positive in query and train image, respectively.
  2. Using `BING` algorithm to extract the landmark from potential loop closure dataset and the directory structure as follows
        ```
        └── UAcampus
          ├── AB
          │   ├── AN
          │   ├── AY
          │   ├── BN
          │   └── BY
          └── AB_txt
              ├── AN
              ├── AY
              ├── BN
              └── BY
        ```
  3. Imporve keypoint matching using a landmark-based image representation in potential loop closure dataset:
        ```bash
        python main.py -dt UAcampus -ks low -tp 630 -fp 10
        ```
     Using the `python main.py -h` to check the meaning of the variables.



