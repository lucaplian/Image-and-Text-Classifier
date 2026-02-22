# Image-and-Text-Classifier
As part of the Machine Learning course. Within this project, we had to implement models in order to classify two image sets and one text set. The project is divided into two main parts:
* Part one, which deals with images, which is further divided into:
    * EDA: it deals with inter-class variability, intra-class variability, class imbalance
    * Model Comparison, we use two models here:
        * MLP: generated suboptimal results, as the flattening operation prevents the model from considering the spatial information
        * CNN: achieved superior results compared to MLP, utilizing pooling layers and 3x3 filters, which are capable of capturing hierarchical features within the images
    * Fine Tuning: used fine tuning on the superior model (CNN), here we used two techniques:
        * Frozen Layers: freezing the backbone. For land patches, the model with frozen layers offered marginally better performance
        * Different learning rates: applied a slow learning rate (3e-6) for the backbone, and a normal rate (3e-4) for the classifier. For imagebits, this model offered preferable results.
* Part two, which deals with sentiment analysis for the text, which is further divided into:
    * EDA: focused on class imbalance and text length analysis
    * Preprocessing and tokenization pipeline:
        * cleaned and normalized the data
        * tokenization of the data: used spacy in order to tokenize the data, each word has an unique index, 0 is reserved for padding and 1 for out-of-vocabulary (OOV) tokens (not in the train dataset)
        * the usage of an pretrained embedding layer (FastText), which represents words as 300-element vectors
        * sequence padding, we made them all to have the global maximum length
        * the embedding matrix, here we transformed the words from our sets into embeddings, and represented unknown words, with a 300-element vector which had random values ranging from -1 to 1
    * Model Comparison, we use three models:
        * RNN, which provided a suboptimal performance, even though it was decent, this is due to not having a capacity to learn long-term dependencies
        * Unidirectional LSTM, which provided the best results, paradoxically, this is may due to having less parameters, thus less prone to overfitting
        * Bidirectional LSTM, which provided slightly worse results than the Unidirectional model.
Project structure:
* hw2_imagebits.ipynb - the code which handles classification of the images from Imagebits dataset
* hw2_land_patches.ipynb - the code which handles classification of the images from Land Patches dataset
* hw2_nlp.ipynb - the code which handles sentiment analysis for the Ro-Sent text.
* hw2_luca_plian_raport.ipynb - report that focuses on detailing the project's implementation and performance analysis
