# Plastic or Glass? - Image Classification with Deep Learning
###### METIS Data Science and Machine Learning Bootcamp 2022 by Krystian Krystkowiak
###### project/month(6/7) focus: DEEP LEARNING
#### Code - cleaning/preprocessing - [GitHub](https://github.com/Krystkowiakk/Plastic-or-Glass---Image-Classification-with-Deep-Learning/blob/main/1.%20Krystkowiak_Krystian_Project_5_Plastic%20or%20Glass%20-%20Image%20Classification%20with%20Deep%20Learning%20-%20image%20loading.ipynb)
#### Code - [GitHub](https://github.com/Krystkowiakk/Plastic-or-Glass---Image-Classification-with-Deep-Learning/blob/main/2.%20Krystkowiak_Krystian_Project_5_Plastic%20or%20Glass%20-%20Image%20Classification%20with%20Deep%20Learning.ipynb)
#### Presentation [GitHub](https://github.com/Krystkowiakk/Plastic-or-Glass---Image-Classification-with-Deep-Learning/blob/main/Project%20Presentation/Krystkowiak_Krystian_Project_5_Plastic%20or%20Glass%20-%20Image%20Classification%20with%20Deep%20Learning.pdf)

ABSTRACT

- Developed a deep learning model to classify glass and plastic waste photos (Kaggle
competition dataset) using Tensor Flow and Keras, testing various models including Neural Networks, Convolutional Neural Networks, and Transfer learning with pre-trained models such as VGG16, VGG19, EfficientNetV2L and Xception.
- This project aims to build a deep learning model that can classify glass and plastic waste photos. The goal is to achieve an acceptable F1 score and potentially integrate the model into more complex systems for use in modern sorting technologies. The dataset used for this project is "Images dataset for classifying household garbage" from Kaggle. The project begins with a logistic regression baseline model, then training, testing, and fine-tunning of different deep learning models. The best performing model, a refined VGG16, is then visualized and proposed as the final solution. The project uses Keynote and Tableau for visualization and presentation.

DESIGN

The project addresses the challenge of efficient waste management by building and improving a machine learning model that can classify different types of waste. Specifically, it focuses on classifying glass and plastic waste, as these materials present more complex classification problems. 

With over 2 billion tons of garbage generated globally every year, mismanagement of waste leads to environmental pollution and human health issues. Innovations such as smart waste containers and self-learning sorting technologies can help to mitigate these problems. Building and improving a machine learning model for waste classification is one step towards creating these solutions.

DATA

The dataset used for this project is part of "Images dataset for classifying household garbage" from Kaggle, created by Mostafa Mohamed. The dataset is available at this link: https://www.kaggle.com/datasets/mostafaabla/garbage-classification and is a mixture of older Kaggle "Garbage Classification" datasets and web scraped images.

It contains 2876 RGB images, belonging to 4 classes: plastic (865 images), green-glass (629 images), brown-glass (607 images) and white-glass (775 images). 

ALGORITHMS

- This project uses various algorithms for data cleaning, loading, preprocessing, and model building. The F1 score is used as the metric to evaluate the models. The data is split into 80% (2300 images) for training and 20% (576 images) for validation. The project includes fine-tuning the models, designing layers, using different optimizers, and implementing regularization techniques.

The following models were trained and tested:

- Logistic Regression (baseline model): F1 on validation set: 0.714
- Neural Network model: F1 on validation set: 0.802
- Convolutional Neural Network (CNN): F1 on validation set: 0.873
- Transfer learning - VGG16: F1 on validation set: 0.879
- Transfer learning - VGG19: F1 on validation set: 0.860
- Transfer learning - EfficientNetV2L (how is it trained matters): F1 on validation set: 0.279
- Transfer learning - Xception: F1 on validation set: 0.846

The final model chosen for this project was VGG16 as it presented slightly better F1 score and balanced other characteristics although CNN and VGG19 presented similar performance and may be the model of choice depending on fine-tuning.

TOOLS

- Python, Pandas, and Numpy for data processing
- Scikit-learn, Keras, Tensorflow for modeling (Logistic Regression, Neural Network model, Convolutional Neural Network (CNN), Transfer learning using various pre-trained models: VGG16, VGG19, EfficientNetV2L and Xception)
- Google Colab to test some of the models
- Seaborn and Tableau for visualization

COMMUNICATION

The final results of the project are presented in a 5-minute recorded video that includes visualizations to communicate the findings and the best performing model.

![Plastic or Glass? - Image Classication with Deep Learning](files/cover.jpg)


