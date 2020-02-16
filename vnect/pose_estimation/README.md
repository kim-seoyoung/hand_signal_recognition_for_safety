# 3D pose estimation
## VNect
* [github](https://github.com/XinArkh/VNect)
* 사람을 중심으로 bounding box만들고 crop -> CNN Regression(2D, 3D joint 추출) -> joint기반으로 skeleton fitting

## spline interpolation
* 분류 모델에 사용하기 위해 frame 수를 100으로 맞추기 위해 사용
* (설명)

## Environments
- python 3.5
  - opencv-python 3.4.4.19
  - tensorflow-gpu 1.12.0 (CUDA 9.0)
  - pycaffe
  - matplotlib 3.0.0 or 3.0.2
  - imagemagick (apt-get)
 
## Setup
```bash
~$ git clone https://github.com/kim-seoyoung/hand_signal_recognition_for_safety
~$ cd hand_signal_recognition_for_safety/vnect/pose_estimation
```

## Usage
###### - numpy파일로 저장
```bash
~$ python3 run_estimator_ps.py --input {input_file} --output-dir {output_directory}
```
###### - numpy파일로 저장 + 3D plot을 gif로 저장
```bash
~$ python3 run_estimator_ps.py --input {input_file} --output-dir {output_directory} --savegif True
```


- ~~(도커 이미지 올리기)~~
