# COMP3000
 
Mindscape is a research tool based around analysis of multi-modal neural networks. Mindscape allows the user to design, train, and compare performance metrics of classification models for identifying stages of dementia with user-selected Convolutional Neural Network architectures and numerical features.

features which can be added or removed from input includes vitals data such as blood pressure and weight, psychological evaluation scores potentially exhibiting behavioural symptoms e.g.  memory loss, and demographic information e.g. age, sex.

the user also controls which image collection to use, each containing 3D MRI volumes. these images are processed through a 3D Convolutional Neural Network architecture, of which there are 3 implementations for the user to choose from; VGG-16, ResNet, and UNet. once the image is processed through the 3D CNN and the numerical features are fed into a series of dense layers, the outputs are flattened and concatenated before sending the result through additional dense layers and classifying into a stage of brain health, such as CN (cognitively normal), MCI (mild cognitive impairment), and AD (Alzheimer's Dementia).

This software can be applied for educational purposes, such as creating figures to illustrate the impact different changes to a model has on the behaviour, or for students to easily experiment and observe machine learning principles through a practical medium, without the need to implement it themselves.

the analytics provided by Mindscape can demonstrate benefits and drawbacks of different dataset or network configurations, these considerations then potentially influencing design decisions in the user's implementation of their own AI system.
