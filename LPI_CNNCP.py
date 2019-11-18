# -*- coding: utf-8 -*-
"""
Created on Fri Nov 23 14:38:21 2018

@author: XIXI
"""
from keras.layers import Input, Dense, Convolution2D, MaxPooling2D, concatenate, Dropout, BatchNormalization,GlobalMaxPooling2D
from keras import regularizers, optimizers,callbacks
from keras.utils.np_utils import to_categorical
from keras.models import Model
from collections import defaultdict
from sklearn.metrics import roc_curve,auc,precision_recall_curve
from LPI_feature import data_two_three_preprocess
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from keras.models import load_model
# import matplotlib.pyplot as plt
import numpy as np
from sklearn.svm import SVC
import keras.backend as K
from keras.callbacks import LearningRateScheduler

import os
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF
from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(2)

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
config = tf.ConfigProto(allow_soft_placement=True)
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.1)
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

KTF.set_session(sess)

def transfer_label_from_prob(proba):
    label = [1 if val >= 0.5 else 0 for val in proba]
    return label


def list_duplicates(seq):
    tally = defaultdict(list)
    for i, item in enumerate(seq):
        tally[item].append(i)
    return ((key, locs) for key, locs in tally.items() if len(locs) > 0)

def calculate_auPR(real_label,prob_list):
    precision,recall,thresholds=precision_recall_curve(real_label,prob_list)
    auPR=auc(recall,precision)
    return auPR

def calculate_roc_auc(real_label,prob_list):
    fpr1,tpr1,thresholds=roc_curve(real_label,prob_list)
    roc_auc=auc(fpr1,tpr1)
    return roc_auc

def scheduler(epoch):
    if epoch %10==0 and epoch !=0:
        lr =K.get_value(model.optimizer.lr)
        K.set_value(model.optimizer.lr,lr*0.1)
        print('lr changed to {}'.format(lr*0.1))
    return K.get_value(model.optimizer.lr)



def calculate_performace(test_num, pred_y, labels):
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    for index in range(test_num):
        if labels[index] == 1:
            if labels[index] == pred_y[index]:
                tp = tp + 1
            else:
                fn = fn + 1
        else:
            if labels[index] == pred_y[index]:
                tn = tn + 1
            else:
                fp = fp + 1

    acc = float(tp + tn) / test_num
    sensitivity = float(tp) / (tp + fn)
    specificity = float(tn) / (tn + fp)
    MCC = float(tp * tn - fp * fn) / (np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
    return acc, sensitivity, specificity, MCC

def calculate_independent_performace(test_num, pred_y, labels):
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    for index in range(test_num):
        if labels[index] == 1:
            if labels[index] == pred_y[index]:
                tp = tp + 1
            else:
                fn = fn + 1
        else:
            if labels[index] == pred_y[index]:
                tn = tn + 1
            else:
                fp = fp + 1

    acc = float(tp + tn) / test_num
    correct_amount=tp + tn
    return acc, correct_amount
# Easy to process data files whose sequence length dose not exceed a fixed range
def Without_crop_LPI():
    filters1 = 96
    kernel_size1 = (49, 15)
    strides1 = (1, 1)
    dropout1 = 0.3
    filters2 = 96
    kernel_size2 = (64, 21)
    strides2 = (1, 3)
    dropout2 = 0.3
    connected = 64
    dropout = 0.2
    epochs = 80
    batch_size = 128
    # Select the fixed length of sequences lncRNA and protein
    len_pro = 1000
    len_lnc = 3500
    copy = False
    train_file='./RPI1446.txt'
    x_data1, x_data2, x_label = data_two_three_preprocess(train_file,len_pro,len_lnc,copy)

    x_label_1=x_label[:,1]
    acc_test1 = []
    acc_test2 = []
    test1_performance = []
    test2_performance = []
    num_cross_val = 10
    i=0
    #skf0=StratifiedKFold(x_label[:,0],n_folds=num_cross_val,random_state=333,shuffle=True)
    skf0=StratifiedKFold(n_splits=num_cross_val,random_state=333,shuffle=True)
    for train_index,test_index in skf0.split(x_data1,x_label_1):
        train1=x_data1[train_index, :]
        test1= x_data1[test_index, :]
        train2=x_data2[train_index, :]
        test2= x_data2[test_index, :]
        train_label1=x_label_1[train_index]
        test_label1=x_label_1[test_index]
        train_label = to_categorical(train_label1, num_classes=2)
        test_label=to_categorical(test_label1, num_classes=2)

        print ('the fold is:',i)

        # if i>0:
        #     break
        i = i + 1
        real_labels = []
        for val in test_label:
            if val[0] == 1:
                real_labels.append(0)
            else:
                real_labels.append(1)
        # with open('./test'+str(i)+'postive_label.txt','w') as f1:
        #     for line in real_labels :
        #         f1.write(str(line)+'\n')
        train_label_new = []
        for val in train_label:
            if val[0] == 1:
                train_label_new.append(0)
            else:
                train_label_new.append(1)

        protein = Input(shape=(train1.shape[1], train1.shape[2], 1))
        lncRNA = Input(shape=(train2.shape[1], train2.shape[2], 1))


        pro=Convolution2D(filters1,kernel_size1,strides=strides1,activation='relu',padding='VALID',use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=regularizers.l2(0.001))(protein)
        pro=BatchNormalization(epsilon=1e-06, momentum=0.9)(pro)
        pro=GlobalMaxPooling2D()(pro)
        #pro=GlobalAveragePooling2D()(pro)
        pro=Dropout(dropout1)(pro)
        #pro=Flatten()(pro)
        print('pro.shape',pro.shape)

        lnc=Convolution2D(filters2,kernel_size2,strides=strides2,activation='relu',padding='VALID',use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=regularizers.l2(0.001))(lncRNA)
        lnc=BatchNormalization(epsilon=1e-06, momentum=0.9)(lnc)
        lnc=GlobalMaxPooling2D()(lnc)
        #lnc=GlobalAveragePooling2D()(lnc)
        lnc=Dropout(dropout2)(lnc)
        #lnc=Flatten()(lnc)
        print('lnc.shape',lnc.shape)

        x = concatenate([pro, lnc], axis=1)
        # fully connected layer
        x = Dense(connected, activation='relu', kernel_regularizer=regularizers.l2(0.001),name='concatenate_layer')(x)
        x = Dropout(dropout)(x)
        # x=Flatten()(x)
        print('x.shape', x.shape)
        main_output = Dense(2, activation='softmax')(x)
        #sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        model = Model(inputs=[protein, lncRNA], outputs=main_output)
        #'rmsprop''adadelta'
        model.compile(optimizer= 'adam', loss='categorical_crossentropy', metrics=['accuracy'])
        reduce_lr=LearningRateScheduler(scheduler)
        #history = model.fit([train1, train2], train_label, epochs=35, batch_size=128,callbacks=[reduce_lr],verbose=2)
        earlystopping = callbacks.EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='auto')
        #history = model.fit([train1, train2], train_label, epochs=epochs, batch_size=batch_size,validation_data=([test1, test2], test_label), shuffle=True, callbacks=[earlystopping],verbose=2)
        history = model.fit([train1, train2], train_label, epochs=epochs, batch_size=batch_size, verbose=2)

        # print(model.summary())
        loss1, accuracy1 = model.evaluate([test1, test2], test_label)
        acc_test1.append(accuracy1)
        test1_predict_prob = model.predict([test1, test2])
        test1_postive_prob=model.predict([test1, test2])[:,1]
        test1_predict_label = np.argmax(test1_predict_prob, axis=1)
        acc, sensitivity, specificity, MCC = calculate_performace(len(real_labels), test1_predict_label,real_labels)
        print('acc,sensitivity, specificity, MCC', acc, sensitivity, specificity, MCC)
        print('loss1, accuracy1',loss1, accuracy1)
        test1_performance.append([acc, sensitivity, specificity, MCC])


    print('test1_performance', np.mean(np.array(test1_performance), axis=0))
    print('acc_test1', np.mean(np.array(acc_test1), axis=0))


def With_crop_LPI():
    # Parameter setting
    filters1 = 32
    kernel_size1 = (49, 21)
    strides1 = (1, 1)
    dropout1 = 0.3
    filters2 = 32
    kernel_size2 = (64, 15)
    strides2 = (1, 1)
    dropout2 = 0.3
    connected = 32
    dropout = 0.3
    epochs = 40
    batch_size = 128
    len_pro = 1000
    len_lnc = 3500
    copy = True
    all_cut_performance = []
    all_really_performance = []
    for kk in range(10):
        print('The fold is:', kk)
        train1, train2, train_label = data_two_three_preprocess(
            './RPI2241CUT_10fold/CUT1/RPI2241_' + str(kk) + '_train_cutdata.txt',len_pro,len_lnc,copy)
        test1, test2, test_label = data_two_three_preprocess(
            './RPI2241CUT_10fold/CUT1/RPI2241_' + str(kk) + '_test_cutdata.txt',len_pro,len_lnc,copy)
        real_labels = []
        for val in test_label:
            if val[0] == 1:
                real_labels.append(0)
            else:
                real_labels.append(1)

        # test_label_new = []
        # for val in test_labels:
        #     if val[0] == 1:
        #         test_label_new.append(0)
        #     else:
        #         test_label_new.append(1)
        #
        # test1_label_new = []
        # for val in test1_labels:
        #     if val[0] == 1:
        #         test1_label_new.append(0)
        #     else:
        #         test1_label_new.append(1)

        protein = Input(shape=(train1.shape[1], train1.shape[2], 1))
        lncRNA = Input(shape=(train2.shape[1], train2.shape[2], 1))

        pro = Convolution2D(filters1, kernel_size1, strides=strides1, activation='relu', padding='VALID', use_bias=True,
                            kernel_initializer='glorot_uniform', bias_initializer='zeros',
                            kernel_regularizer=regularizers.l2(0.001))(protein)
        pro = BatchNormalization(epsilon=1e-06, momentum=0.9)(pro)
        pro = GlobalMaxPooling2D()(pro)
        # pro=GlobalAveragePooling2D()(pro)
        pro = Dropout(dropout1)(pro)
        # pro=Flatten()(pro)
        print('pro.shape', pro.shape)

        lnc = Convolution2D(filters2, kernel_size2, strides=strides2, activation='relu', padding='VALID', use_bias=True,
                            kernel_initializer='glorot_uniform', bias_initializer='zeros',
                            kernel_regularizer=regularizers.l2(0.001))(lncRNA)
        lnc = BatchNormalization(epsilon=1e-06, momentum=0.9)(lnc)
        lnc = GlobalMaxPooling2D()(lnc)
        # lnc=GlobalAveragePooling2D()(lnc)
        lnc = Dropout(dropout2)(lnc)
        # lnc=Flatten()(lnc)
        print('lnc.shape', lnc.shape)

        x = concatenate([pro, lnc], axis=1)
        # fully connected layer
        x = Dense(connected, activation='relu', kernel_regularizer=regularizers.l2(0.001), name='concatenate_layer')(x)
        x = Dropout(dropout)(x)
        # x=Flatten()(x)
        print('x.shape', x.shape)
        main_output = Dense(2, activation='softmax')(x)
        # sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        model = Model(inputs=[protein, lncRNA], outputs=main_output)
        # 'rmsprop''adadelta'
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        # history = model.fit([train1, train2], train_label, epochs=35, batch_size=128,callbacks=[reduce_lr],verbose=2)
        #earlystopping = callbacks.EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='auto')
        #history = model.fit([train1, train2], train_label, epochs=epochs, batch_size=batch_size, validation_data= ([test1, test2], test_label),shuffle=True,callbacks=[earlystopping], verbose=2)
        history = model.fit([train1, train2], train_label, epochs=epochs, batch_size=batch_size, shuffle=True, verbose=2)
        # print(model.summary())

        app_test_label = []
        app_test_name = []
        with open('./RPI2241CUT_10fold/CUT1/RPI2241_' + str(kk) + '_testdata.txt','r') as f1:
            data_all = f1.readlines()

        for i in range(len(data_all)):
            data2 = data_all[i].split('$')
            app_test_label.append(int(data2[0]))
            app_test_name.append(data2[0] + '$' + data2[1] + '$' + data2[2])
        # print('app_test_label:',app_test_label)

        data_label_name = []
        with open('./RPI2241CUT_10fold/CUT1/RPI2241_' + str(kk) + '_test_cutdata.txt', 'r')as f2:
            data3 = f2.readlines()
            for j in range(len(data3)):
                data4 = data3[j].split('$')
                data5 = data4[2].split('_')
                data_label_name.append(data4[0] + '$' + data4[1] + '$' + data5[0])
        data_list = list(list_duplicates(data_label_name))
        # print('data_list:',data_list)
        data_index = []
        data_index_dict = {}

        for i in range(len(data_list)):
            data_index_dict[data_list[i][0]] = data_list[i][1]
        for i in range(len(app_test_name)):
            data_index.append(data_index_dict[app_test_name[i]])

        # model.save('DE_newtest1_savemodel.h5')
        loss1, accuracy1 = model.evaluate([test1, test2], test_label)
        print('loss1, accuracy1', loss1, accuracy1)

        test1_predict_prob = model.predict([test1, test2])
        test1_predict_label = np.argmax(test1_predict_prob, axis=1)

        test1_predict_realprob = test1_predict_prob[:, 1]
        # print('test1_predict_label',test1_predict_label)
        # print('test1_predict_realprob:',test1_predict_realprob)
        # print('real_labels:',real_labels)
        test1_predice_trueprob = []
        for i in range(len(data_index)):
            all_prob = []
            for j in range(len(data_index[i])):
                data5 = test1_predict_realprob[data_index[i][j]]
                all_prob.append(data5)
            max_prob = max(all_prob)
            test1_predice_trueprob.append(max_prob)

        proba = transfer_label_from_prob(test1_predice_trueprob)
        # print('proba:',proba)
        print('really performance(without cut):')
        acc1, sensitivity1, specificity1, MCC1 = calculate_performace(len(app_test_label), proba,app_test_label)
        all_really_performance.append([acc1, sensitivity1, specificity1, MCC1])
        mean_really_performance = np.mean(np.array(all_really_performance), axis=0)
        print('acc1, sensitivity1, specificity1, MCC1', acc1, sensitivity1, specificity1, MCC1)
        print('mean_really_performance:', mean_really_performance)

        print('cut performance:')
        acc2, sensitivity2, specificity2, MCC2 = calculate_performace(len(real_labels), test1_predict_label,real_labels)
        all_cut_performance.append([acc2,  sensitivity2, specificity2, MCC2])
        mean_cut_performance = np.mean(np.array(all_cut_performance), axis=0)
        print('acc2, sensitivity2, specificity2, MCC2', acc2, sensitivity2, specificity2, MCC2)
        print('mean_cut_performance:', mean_cut_performance)

def Predict_new_interactions():
    i = 0
    filters1 = 96
    kernel_size1 = (49, 15)
    strides1 = (1, 1)
    dropout1 = 0.3
    filters2 = 96
    kernel_size2 = (64, 21)
    strides2 = (1, 3)
    dropout2 = 0.3
    connected = 64
    dropout = 0.2
    epochs = 80
    batch_size = 128
    len_pro = 1000
    len_lnc = 3500
    copy = True
    train1, train2, train_label = data_two_three_preprocess('./RPI1446.txt', len_pro, len_lnc, copy)

    protein = Input(shape=(train1.shape[1], train1.shape[2], 1))
    lncRNA = Input(shape=(train2.shape[1], train2.shape[2], 1))

    pro = Convolution2D(filters1, kernel_size1, strides=strides1, activation='relu', padding='VALID', use_bias=True,
                        kernel_initializer='glorot_uniform', bias_initializer='zeros',
                        kernel_regularizer=regularizers.l2(0.001))(protein)
    pro = BatchNormalization(epsilon=1e-06, momentum=0.9)(pro)
    pro = GlobalMaxPooling2D()(pro)
    # pro=GlobalAveragePooling2D()(pro)
    pro = Dropout(dropout1)(pro)
    # pro=Flatten()(pro)
    print('pro.shape', pro.shape)

    lnc = Convolution2D(filters2, kernel_size2, strides=strides2, activation='relu', padding='VALID', use_bias=True,
                        kernel_initializer='glorot_uniform', bias_initializer='zeros',
                        kernel_regularizer=regularizers.l2(0.001))(lncRNA)
    lnc = BatchNormalization(epsilon=1e-06, momentum=0.9)(lnc)
    lnc = GlobalMaxPooling2D()(lnc)
    # lnc=GlobalAveragePooling2D()(lnc)
    lnc = Dropout(dropout2)(lnc)
    # lnc=Flatten()(lnc)
    print('lnc.shape', lnc.shape)

    x = concatenate([pro, lnc], axis=1)
    # fully connected layer
    x = Dense(connected, activation='relu', kernel_regularizer=regularizers.l2(0.001), name='concatenate_layer')(x)
    x = Dropout(dropout)(x)
    # x=Flatten()(x)
    print('x.shape', x.shape)
    main_output = Dense(2, activation='softmax')(x)
    # sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model = Model(inputs=[protein, lncRNA], outputs=main_output)
    # 'rmsprop''adadelta'
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    history = model.fit([train1, train2], train_label, epochs=epochs, batch_size=batch_size, shuffle=True, verbose=2)

    model.save('my_independent_model.h5')
    # Processing independent test set data
    test1, test2, test_label = data_two_three_preprocess('./human_independent_data.txt',len_pro,len_lnc,copy)
    np.save('test1', test1)
    np.save('test2', test2)
    np.save('test_label', test_label)

    # test1 = np.load('test1.npy')
    # test2 = np.load('test2.npy')
    # test_label = np.load('test_label.npy')

    i = i + 1
    real_labels = []
    for val in test_label:
        if val[0] == 1:
            real_labels.append(0)
        else:
            real_labels.append(1)

    app_test_label = []
    app_test_name = []
    with open('./human_data.txt', 'r') as f1:
        data_all = f1.readlines()

    for i in range(len(data_all)):
        data2 = data_all[i].split('$')
        app_test_label.append(int(data2[0]))
        app_test_name.append(data2[0] + '$' + data2[1] + '$' + data2[2])
    # print('app_test_label:',app_test_label)

    data_label_name = []
    with open('./human_independent_data.txt','r')as f2:
        data3 = f2.readlines()
        for j in range(len(data3)):
            data4 = data3[j].split('$')
            data5 = data4[2].split('&')
            data_label_name.append(data4[0] + '$' + data4[1] + '$' + data5[0])
    data_list = list(list_duplicates(data_label_name))
    # print('data_list:',data_list)
    data_index = []
    data_index_dict = {}

    for i in range(len(data_list)):
        data_index_dict[data_list[i][0]] = data_list[i][1]
    for i in range(len(app_test_name)):
        data_index.append(data_index_dict[app_test_name[i]])

    # model.save('DE_newtest1_savemodel.h5')
    loss1, accuracy1 = model.evaluate([test1, test2], test_label)
    print('loss1, accuracy1', loss1, accuracy1)

    test1_predict_prob = model.predict([test1, test2])
    with open('./independent_predict.txt','w') as f1:
        for line in test1_predict_prob:
            f1.write(str(line)+'\n')
    test1_predict_label = np.argmax(test1_predict_prob, axis=1)

    test1_predict_realprob = test1_predict_prob[:, 1]
    # print('test1_predict_label',test1_predict_label)
    # print('test1_predict_realprob:',test1_predict_realprob)
    # print('real_labels:',real_labels)
    test1_predice_trueprob = []
    for i in range(len(data_index)):
        all_prob = []
        for j in range(len(data_index[i])):
            data5 = test1_predict_realprob[data_index[i][j]]
            all_prob.append(data5)
        max_prob = max(all_prob)
        test1_predice_trueprob.append(max_prob)

    proba = transfer_label_from_prob(test1_predice_trueprob)
    # print('proba:',proba)
    print('really performance:')
    acc1, true_nums = calculate_independent_performace(len(app_test_label), proba, app_test_label)
    print('acc1', acc1, 'true_nums', true_nums)

    print('cut performance:')
    acc2, cut_true = calculate_independent_performace(len(real_labels), test1_predict_label, real_labels)
    print('acc2', acc2)

def Run_LPI_CNNCP_model(Y_crop_LPI, N_crop_LPI, Independent_test):
    if Y_crop_LPI:
        Without_crop_LPI()

    if N_crop_LPI:
        With_crop_LPI()

    if Independent_test:
        Predict_new_interactions()
if __name__=='__main__':
    Run_LPI_CNNCP_model(Y_crop_LPI=True, N_crop_LPI=False, Independent_test=False)
