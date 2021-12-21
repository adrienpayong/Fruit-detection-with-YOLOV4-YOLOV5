# YOLOv4
The original YOLO algorithm was created by Joseph Redmon, who is also the creator of the Darknet custom framework.
After 5 years of study and development to the third generation of YOLO (YOLOv3), Joseph Redmon announced his retirement from the area of computer vision and the discontinuation of developing the YOLO algorithm due to concerns that his research will be misused in military applications.
He does not, however, challenge the continuance of study by any person or group based on the YOLO algorithm's early principles.

Alexey Bochkovsky, a Russian researcher and engineer who constructed the Darknet framework and three earlier YOLO architectures in C based on Joseph Redmon's theoretical concepts, collaborated with Chien Yao and Hon-Yuan to publish YOLOv4 in April 2020.
(2020, Bochkovskiy) 
## Object detection architecture
Along with the development of YOLO, several object identification algorithms using various methodologies have achieved outstanding results.
Since then, two architectural object detection ideas have emerged: one-stage detector and two-stage detector.

The input image characteristics are compressed down by the feature extractor (Backbone) and then sent to the object detector (containing the Detection Neck and Detection Head), as shown in Figure 15.
Detection Neck (or Neck) functions as a feature aggregation, mixing and combining the features created in the Backbone to prepare for the detection process in the Detection Head (or Head).

The distinction here is that Head is in charge of detection, including localization and classification, for each bounding box.
As shown below,[source](https://github.com/adrienpayong/object-detection/blob/main/Capture1.PNG).
the two-stage detector does these two jobs independently and then aggregates their findings (Sparse Detection), while the one-stage detector performs both tasks simultaneously (Dense Detection) (Solawetz, 2020).
Because YOLO is a one-stage detector, You Only Look Once. 
# Overview on Backbone CSPDarknet53 

The authors investigated three alternatives for the YOLOv4 model's backbone (feature extractor): CSPResNext53, CSPDarknet53, and EfficientNet-B3, the most sophisticated convolutional network at the time.
CSP Darknet53 neural network was judged to be the best optimum model based on theoretical rationale and several testing.


The CSPResNext50 and CSPDarknet53 (CSP stands for Cross Stage Partial) architectures are both developed from the DenseNet design, which takes the prior input and concatenates it with the current input before proceeding into the dense layer (Huang, et al., 2018).
DenseNet was created to link layers in an extremely deep neural network in order to solve vanishing gradient challenges (as ResNet)


To be more specific, each stage of DenseNet is made up of a dense block and a transition layer, and each dense block is made up of k dense layers.
After passing through the dense block, the input will be routed to the transition layer, which will change the size (downsample or upsample) using convolution and pooling.
The ith dense layer's output will be concatenated with its own input to generate the input for the following (i + 1)th layer.
For example, in the first dense layer, the input xo has created the output x1 after progressing through convolutional layers.
The output x1 is then concatenated with its own input x0, and the result of this concatenation becomes the input of the second dense layer. 

The CSP (Cross Stage Partial) is built on the same premise as the DenseNet described above, except that instead of utilizing the full-size input feature map at the base layer, the input will be divided into two halves.
A chunk will be sent via the dense block as normal, while another will be routed directly to the next step without being processed.
As a consequence, various dense layers will learn duplicated gradient information again.
2019 (Wang et al.) 
When these concepts were combined with the Darknet-53 design in YOLOv3, the residual blocks were replaced with dense blocks.
CSP preserves features via propagation, stimulates the network to reuse features, decreases the number of network parameters, and aids in the preservation of fine-grained features for more efficient forwarding to deeper layers.
Given that increasing the number of densely linked convolutional layers may result in a drop in detection speed, only the final convolutional block in the Darknet-53 backbone network that can extract richer semantic features is enhanced to be a dense block. 
