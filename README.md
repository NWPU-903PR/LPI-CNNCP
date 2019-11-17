# LPI-CCH
LPI-CNNCP

Predicting lncRNA-protein interactions with copy-padding trick and convolutional neural network. 

 Dependencies:

(1) numpy; (2)scikit-learn


Sample pair naming format:

(1)Original sample pair：

label+'$'+protein_tag+'$'+lncRNA_tag+'$'+protein_sequence+lncRNA_sequence

(2)Cut sample pairs：

label+'$'+protein_tag+'$'+lncRNA_tag+'_(subsequence index)'+'$'+protein_subsequence+lncRNA_subsequence

Usage:

(1)perform 10 fold cross validation(without cut):

Run_LPI_CNNCP_model(Y_crop_LPI=True, N_crop_LPI=False, Independent=False)

(2)perform 10 fold cross validation(with cut):

Run_LPI_CNNCP_model(Y_crop_LPI=False, N_crop_LPI=True, Independent=False)

(3)predict a new lncRNA-protein pair

Run_LPI_CNNCP_model(Y_crop_LPI=False, N_crop_LPI=False, Independent=True)
