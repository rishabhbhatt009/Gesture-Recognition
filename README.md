# Gesture-Recognition
Computer Vision Project 

### Problem Statement 
**Task identification** : Given a video stream as input the task is to detect and predict the sequence of signs and gestures that are performed in order to generate text/speech in natural language.

### Challenges
Sign language is a visual language performed with dynamic movement of hand gestures, body posture, and facial expression. SLR has the following challenges : 

<ul>
    <li>Sign languages have a diverse and constantly evolving vocabulary which makes it very hard to create a generalized dataset</li>
    <li>Sings are of two types - Static signs and Dynamic gestures</li>
    <li>Existing Datasets do not provide a large-scale vocabulary of signs</li>
    <li>It is hard to detect the object and isolate the ROI (Region of interest) in the wild</li>
    <li>Leverage minimal hardware requirement – only using monocular RGB based videos</li>
</ul>

### Project Workflow 
![Project Work Flow](/doc_images/Work_Flow.jpg)

### Dataset Selection
<ul>
    <li>WLASL : a largescale dataset containing monocular RGB based videos for Word-Level ASL, containing more than 2000 words performed by over 100 signers</li>
    <li>SHREC : dataset contains 14 dynamic gestures performed by 28 participants (all participants are right-handed) and captured by the Intel RealSense short range depth camera. Each gesture is performed between 1 to 10 times by each participant</li>
    <li>MS-ASL : A Large-Scale Data Set and Benchmark for Understanding American Sign Language by Microsoft, that covers over 200 signers, signer independent sets, challenging and unconstrained recording conditions and a large class count of 1000 signs</li>
    <li>Custom Dataset : Self collection of RGB based videos through webcams</li>
</ul>

### Pre-Processing Module
<ul>
    <li>Frame Sampling</li>
    <li>Media-pipe Pipeline –</li>
        <ul>
            <li>Palm Detection</li>
            <li>Hand Isolation </li>
            <li>Landmark Detection</li>
        </ul>
    </li>
    <li>Key-Point Extraction –</li>
        <ul>
            <li>33 pose landmarks</li>
            <li>21 hand landmarks x 2 (left and right)</li>
            <li>468 facial landmarks</li>
        </ul>
</ul>

Final out frames from the pre-processing module. 

![Keypoint Extraction](/doc_images/Pre_Proc.png)

### Skeleton Based LSTM Model Architecture
![img_5.png](/doc_images/LSTM%20Model.jpg)

### Dataset-1 WLASL2000 
<ul>
    <li>Word-Level dataset containing 21k RGB based videos performed by 119 signers</li>
    <li>The diversity of signers provides inter-signer variations, which facilitates the generalization ability of the trained sign recognition models.</li>
    <li>Apart from providing a gloss label for all videos, meta information Body Bounding Box, Temporal Boundary and Signer Diversity is also included in the dataset.</li>
</ul>

#### WLASL2000 Stats :

![WLASL Stats.png](/doc_images/WLASL_Stats.png)

### Dataset-2 Custom Dataset
4 signers X 25 signs / person X 30 samples / sign = 120 videos / sign = 3000 videos

![Custom Dataset](/doc_images/img_DB2.png)

### Baseline Results and Findings
Findings from initial analysis show very low accuracy score when baseline models are trained on WLASL2000. Even our skeleton-based LSTM model doesn’t produce significant results.  

![Results](/doc_images/Results.JPG)

### Results of Exploratory Analysis of WLASL2000 dataset

Possible reasons for low accuracy :
<ul>
    <li>Videos have been removed from their initial source therefore the model is trained on a smaller dataset</li>
    <li>We observed that few signs have different gestures, which could be attributed to different dialects based on geographical regions.</li>
</ul>





