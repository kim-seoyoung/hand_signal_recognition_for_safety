# Hand_Signal_Recognition_for_Safety(HSRS)

![image](https://user-images.githubusercontent.com/39910353/73731602-1dd16380-477c-11ea-952e-12eb50a031dd.png)

## 프로젝트 소개
* **Hand_Signal_Recognition_for_Safety(HSRS)** 는 사람의 수신호를 감지 및 분류 모델 개발 프로젝트이다.
* **총 네 개의 step**으로 이루어져 있으며 서로 다른 **두개의 Back bone 모델**을 사용하여 두가지 버전으로 개발되고 있다.
* **verstion 1**: VNect base model([link](https://github.com/kim-seoyoung/hand_signal_recognition_for_safety/tree/master/vnect)),    **version 2** : VideoPose3D base model([link]())

## Step1 : Prepareing training Data

* **UTD Multimodal Human Action Dataset** ([link](https://personal.utdallas.edu/~kehtar/UTD-MHAD.html))
  - Microsoft Kinect sensor와 inertial sensor를 각각 하나씩 이용
  - 8명의 사람의 27가지의 action으로 구성
  - RGB videos, depth videos, skeleton joint positions, inertial sensor signal이 포함되어 있음
  ![image](https://user-images.githubusercontent.com/39910353/73733880-18761800-4780-11ea-807c-ba61cc2f3e5d.png)
  - Draw X, Draw circle 등 실제 사용 수신호와 유사한 data이용 
  - Kinect v2이용한 추가적인 dataset존재
  
### Signal 참고

* **ISO 16715 크레인 안전 수신호**

![image](https://user-images.githubusercontent.com/39910353/73734042-5c691d00-4780-11ea-844a-cd49c1ceb132.png)

* **덤프트럭 수신호**

![image](https://user-images.githubusercontent.com/39910353/73734074-6a1ea280-4780-11ea-91be-1f5aa6698a03.png)


## Step2 : Building Pose Estimation model

### VideoPose3D model
* [VideoPose3D](https://github.com/facebookresearch/VideoPose3D)
* 동영상의 각 프레임으로 부터 2D 관절을 먼저 추출한 후 2D관절의 연속성에 따라 3D 관절을 추출해 내는 모델로 17개 관절을 추출한다.
* 2D 관절을 추출하고 3D 관절을 추출하는 2가지 과정을 거친다. 이때 2D 관절은 'Detectron'모델을 사용한다


### VNect model
* [VNect_pose_estimaion](https://github.com/kim-seoyoung/bestsafe/tree/master/pose_estimation)





## Step3 : Data Preprocessing

* training data를 Pose estimation model을 사용하여 관절 점을 추출
* spline interpolation 기법을 활용하여 프레임 수 조정


## Step4 : Building Classification model

*Two-Stream Adaptive Graph Convolutional Networks을  Back bone model로한 분류 모델 구축
![image](https://user-images.githubusercontent.com/39910353/73735165-2c227e00-4782-11ea-999e-c882b659b5cd.png)
