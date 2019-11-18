# LPI-CNNCP


LPI-CNNCP designs a copy-padding trick to turn the protein/RNA sequences with variable-length into the fixed-length sequences, letting them meet the input requirements of CNN model. Then, the high-order one-hot encoding way is used to transform the protein/RNA sequences into image-like inputs of non-independent for capturing the dependencies among amino acids (or nucleotides). In the end, these encoded protein/RNA sequences are feed into a convolutional neural network to predict the lncRNA-protein interactions.


## System Requirements

The LPI-CNNCP is supported on Linux operating system python 3, Keras version=2.2.4 and its backend is TensorFlow Sklearn, scikit-learn version=0.21.3, numpy version=1.17.4

## Content
./LPI_CNNCP.py: the python code, it can be ran to reproduce our results.

./LPI_feature.py: the python code, it can be used to generate feature coding matrices.

./data.zip: the training, testing and independent testing dataset with sequence, pair name and label.


## Users's Guide
File Example is an example of a program input file. The user needs to process the experimental data into a sample format in the Example file.

（1）If the sequence length of all experiment data is within a fixed length range, it can be processed according to file ./Example/Example_RPI1446 for convenient operation.

Sample pair naming format:

label+'$'+protein_tag+'$'+lncRNA_tag+'$'+protein_sequence+'#'+lncRNA_sequence.

Example：
1$3UZK-3$3UZK-A$MAHKKGLGSTRNG#GGUCAAGAUGGUA

（2）If the sequence length of the experiment data exceeds the fixed length range, it can be processed according to file ./Example/Example_RPI2241.

Sample pair naming format:

label+'$'+protein_tag+'$'+lncRNA_tag+'_(subsequence index)'+'$'+protein_subsequence+'#'+lncRNA_subsequence

Example：
1$2B63-B$2B63-R_1$MSDLANSEKYYDEDPYGFEDESAPITAE#CAGCACUGAUUGCGGUCGAGGUAGCUUGAUG

## Execute Step
1.Configure the hyperparameters(e.g. the filter number, filter size, pooling size, the neuron number of fully connected layer, strides and Dropout) and process data files as required.

2.Run LPI-CNNCP by configuring the corresponding parameters in function Run_LPI_CNNCP_model in the LPI-CNNCP.py program file as needed.

(1)predict a new lncRNA-protein pair

Configuring Run_LPI_CNNCP_model(Y_crop_LPI=False, N_crop_LPI=False, Independent=True), then run LPI-CNNCP,the final predicted probability values are written into text formats(independent_predict.txt).

(2)Evaluation model prediction performance

1)perform 10 fold cross validation(without cut):

Configuring Run_LPI_CNNCP_model(Y_crop_LPI=True, N_crop_LPI=False, Independent=False), then run LPI-CNNCP, the final evaluation metrics will be output.

2)perform 10 fold cross validation(with cut):

Configuring Run_LPI_CNNCP_model(Y_crop_LPI=False, N_crop_LPI=True, Independent=False), then run LPI-CNNCP, the final evaluation metrics will be output.


