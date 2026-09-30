# Project Interview Responses

**1. Problem and Motivation**
There are many projects related to the pothole detection.. but no project is really solving an issue.. all are academic projects.. just focusing on showing that thy can build.. but no real use case.. My project is a part of reaal-time pothole detection software.. I'm haandling the ML part of it.. this is not the whole project but the part of it.. my main contribution is sevrity prediction along with the pothole detection.. which helps the software .. well I'm sharing the problem statement:
Practise project: AI-Powered Road Damage Detection & Municipal Management System.
This is a good learning project because you can start with a relatively simple ML task and gradually turn it into a complete SIH-style software system.
The system will allow someone to upload/take a road image, and AI will automatically identify the damage, predicting Pothole, Confidence, and Severity, which feeds into an Admin dashboard.

**2. What changed in each phase and why (Failures and surprises)**
Phase 1 failed because of low recall (0.6627) and poor localization accuracy under strict IoU (mAP@50:95 of 0.4941). The baseline model was overly conservative. Diagnosis concluded that 665 images provide insufficient visual and environmental diversity. Phase 2 required expanding the training dataset to expose the model to varying road conditions, and replacing bounding-box area calculations with a YOLOv8-Seg instance segmentation model because bounding boxes include extraneous background road pixels that distort geometric calculations.

**3. Core contribution vs off-the-shelf**
core contribution is.. not inventing something.. but the ubique integration of different parts together and working something real on depth part..

**4. Dataset sources, licenses, splits**
second phase.. we merged 4 different datasets...how I merged them..
The initial dataset contains 665 PNG images paired with Pascal VOC XML annotations, split into 80% training, 10% validation, and 10% testing subsets.
To address low recall, the Pothole Detection Dataset-v2-2022 was identified (2,083 images), parsed, and a randomized 70% training, 15% validation, and 15% testing split was performed.
The Road Damage Dataset 2022 (RDD2022) was also evaluated, extracting D40 annotations.
Merging datasets requires strict deduplication to ensure identical images do not leak into testing.

**5. Which results and graphs are final**
Yes, the files currently in `context/` are the final results.

**6. Which sample images are safe to publish**
Yes, the images in `pothole_yolo/images/test/` are safe to publish.

**7. Target labs / Audience**
research labs on conmputer vision
