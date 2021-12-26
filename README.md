# Overview of YOLOv4
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
As shown below,![source](https://github.com/adrienpayong/object-detection/blob/main/Capture1.PNG).

the two-stage detector does these two jobs independently and then aggregates their findings (Sparse Detection), while the one-stage detector performs both tasks simultaneously (Dense Detection) (Solawetz, 2020).
Because YOLO is a one-stage detector, You Only Look Once. 
# Backbone CSPDarknet53 

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
### Neck – Additional block – SPP block

The output feature maps of the CSPDarknet53 backbone were delivered to an extra block (Spatial Pyramid Pooling block) to extend the receptive field and separate out the most relevant features before sending to feature aggregation architecture in the neck.

Many CNN-based (convolutional neural network) models have fully connected layers that take only images of certain dimensions as input.
SPP was created with the goal of producing a fixed-size output regardless of the size of the input.
SPP also aids in the extraction of essential characteristics by pooling multi-scale versions of itself.
Since fully connected layers were removed from YOLOv2, the YOLO algorithm has evolved into an FCN-based (fully convolution network) model that accepts images of varying dimensions as input. 

Furthermore, YOLO must forecast and localize the locations of the bounding boxes based on the 𝑆 × S grid cell displayed on the image.
As a result, turning two-dimensional feature maps into a fixed-size one-dimensional vector isn't always a good idea.
As a result, the SPP block has been updated to maintain the output spatial dimension.
The new SPP block was situated near the backbone.

A 1X1 convolution is employed between the backbone and the SPP block to reduce the amount of input feature maps delivered to the SPP block .
Following that, the input feature maps are duplicated and pooled in multiple scales using the same approach as the traditional SPP block, except that padding is employed to preserve the output feature maps a consistent size, such that three feature maps remain the sizes of 𝑠𝑖𝑧𝑒𝑓𝑚𝑎𝑝 × 𝑠𝑖𝑧𝑒𝑓𝑚𝑎𝑝 × 512.. 
### Neck – Feature Aggregation – PANet
After passing through the backbone, the input image's features are converted into semantical features (or learned features).
In other words, as the input image progresses through the low-level layers, the intricacy of semantical characteristics increases but the spatial resolution of feature maps diminishes owing to downsampling.
As a result, spatial information and fine-grained characteristics are lost.
For YOLOv3's neck, Joseph Redmon used the Feature Pyramid Network (FPN) architecture to maintain these fine-grained characteristics.
The FPN design used a top-down approach to transmit semantical information (from the high-level layer) and then concatenate them to fine-grained features (from the backbone's low-level layer) for predicting tiny objects in the large-scale detector.
Path Aggregation Network (PAN) is a more advanced version of FPN.

Because the flow in FPN architecture is top-down, only the large-scale detector from low-level layers in FPN may receive semantic information from high-level layers and fine-grained features from low-level layers in the lateral backbone at the same time .
Currently, the small-scale detector from high-level layers in FPN detects objects using solely semantic information.
The notion of concatenating semantic features with fine-grained features at high-level layers was suggested to increase the performance of the small and medium-scale detectors.

![source](https://github.com/adrienpayong/object-detection/blob/main/path.PNG)

However, the backbone of today's deep neural networks comprises a large number of layers (can be more than 100 layers).
As a result, in FPN, fine-grained features must traverse a lengthy trip from low-level to high-level layers.
The PAN architecture's developers offered a bottom-up augmentation approach in addition to the top-down one utilized in FPN.
As a result, a "shortcut" was built to link fine-grained characteristics from lower-level layers to higher-level layers.
This "shortcut" has less than ten layers, allowing for easier information flow (Liu et al., 2018). 
## Adaptive Feature Pooling Structure 
Previously utilized algorithms, such as the Mask-RCNN, employed information from a single stage to forecast masks.
If the area of interest was wide, it employed ROI Align Pooling to pull features from higher levels.
Although fairly precise, this might nevertheless result in undesirable outcomes since two proposals with as little as 10-pixel differences can be allocated to two separate layers, despite the fact that they are relatively identical proposals.

To circumvent this, PANet employs features from all tiers and allows the network to choose which are valuable.
To extract the features for the object, it runs the ROI Align operation on each feature map.
Following this, an element-wise max fusion operation is performed to allow the network to adapt to new features.

![source](https://github.com/adrienpayong/object-detection/blob/main/Capture7.PNG)

### Fully-connected Fusion

Fully Convolutional Network (FCN) is utilized instead of fully connected layers in Mask-RCNN because it retains spatial information while reducing the number of parameters in the network.
However, since the parameters are shared across all spatial places, the model does not learn how to utilize pixel locations for prediction; it will, by default, depict sky at the top of the image and roads at the bottom.
Fully-connected layers, on the other hand, are sensitive to location and may adapt to various spatial locations. PANet makes advantage of information from both layers to produce a more accurate mask prediction. 

### Head – YOLOv3
The job of the head in a one-stage detector is to make dense predictions.
The dense prediction is the final prediction, which is made up of a vector that contains the predicted bounding box coordinates (center, height, breadth), the prediction confidence score, and the probability classes.
For detection, YOLOv4 uses the same head as YOLOv3 with anchor-based detection stages and three degrees of detection granularity (Solawetz, 2020) 
