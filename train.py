from Bio import SeqIO
import numpy as np
import matplotlib.pyplot as plt
import json

import tensorflow as tf
from tensorflow import keras
from keras.layers import Activation
from keras.layers import Dropout
from keras.layers import Dense
from keras.layers import Input
from keras.layers import Flatten
from keras.layers.merge import concatenate
from keras.models import Sequential, load_model, save_model, Model
from keras.optimizers import Adam, Adagrad
from keras.layers.convolutional import Conv1D, MaxPooling1D


def make_binary_for_3_mers(index):
    '''
    input: index of a 3-mer 
    output: a numbpy array of shape (1, 13)
    '''
    bin_array = np.zeros((1, 13))
    p = index
    i = 12
    while p:
        r = p % 2
        bin_array[0][i] = r
        p = p // 2
        i -= 1
    return bin_array

def find_percentage(t, positive_proteins, negative_proteins):
    c = 0
    for i in negative_proteins:
        if len(i.seq) < t:
            c+=1
    for i in positive_proteins:
        if len(i.seq) < t:
            c+=1
    return  ( c/(len(negative_proteins)+len(positive_proteins)) ) * 100

def create_1_mer_one_hot_vectors(positive_proteins, negative_proteins):
    L = 350 # we are using 350 as length of a row in the 3d input matrix as 95% capsid and non-capsid proteins have length less than 350
    X_1_mer_one_hot_vectors = []
    for i in range(len(negative_proteins)):
        neg_protein = negative_proteins[i].seq
        
        if len(neg_protein) <= L:
            tmp_1_mer_one_hot_vector = np.zeros((1, L, 20)) # 1-mer one hot vetors, for neg_protein
            for j in range(len(neg_protein)):
                x = np.zeros((1, 20))
                x[0][kmers_index[neg_protein[j]]] = 1
                tmp_1_mer_one_hot_vector[0][j] = x
            X_1_mer_one_hot_vectors.append(tmp_1_mer_one_hot_vector)
            Y_train.append(0)
        else:
            number_of_chunks = len(neg_protein) // L
            remainder_chunk = len(neg_protein) % L
            start = 0
            end = L
            for k in range(number_of_chunks):
                chunk = neg_protein[start:end]
                tmp_1_mer_one_hot_vector = np.zeros((1, L, 20)) # 1-mer one hot vetors, for chunk
                pos = 0
                for j in chunk:
                    x = np.zeros((1, 20))
                    x[0][kmers_index[j]] = 1
                    tmp_1_mer_one_hot_vector[0][pos] = x
                    pos += 1
                X_1_mer_one_hot_vectors.append(tmp_1_mer_one_hot_vector)
                Y_train.append(0)
                start = end
                end = start + L
            if remainder_chunk:
                chunk = neg_protein[start:start+remainder_chunk]
                tmp_1_mer_one_hot_vector = np.zeros((1, L, 20)) # 1-mer one hot vetors, for chunk
                pos = 0
                for j in chunk:
                    x = np.zeros((1, 20))
                    x[0][kmers_index[j]] = 1
                    tmp_1_mer_one_hot_vector[0][pos] = x
                    pos += 1
                X_1_mer_one_hot_vectors.append(tmp_1_mer_one_hot_vector)
                Y_train.append(0)

    for i in range(len(positive_proteins)):
        post_protein = positive_proteins[i].seq
        
        if len(post_protein) <= L:
            tmp_1_mer_one_hot_vector = np.zeros((1, L, 20)) # 1-mer one hot vetors, for neg_protein
            for j in range(len(post_protein)):
                x = np.zeros((1, 20))
                x[0][kmers_index[post_protein[j]]] = 1
                tmp_1_mer_one_hot_vector[0][j] = x
            X_1_mer_one_hot_vectors.append(tmp_1_mer_one_hot_vector)
            Y_train.append(1)
        else:
            number_of_chunks = len(post_protein) // L
            remainder_chunk = len(post_protein) % L
            start = 0
            end = L
            for k in range(number_of_chunks):
                chunk = post_protein[start:end]
                tmp_1_mer_one_hot_vector = np.zeros((1, L, 20)) # 1-mer one hot vetors, for chunk
                pos = 0
                for j in chunk:
                    x = np.zeros((1, 20))
                    x[0][kmers_index[j]] = 1
                    tmp_1_mer_one_hot_vector[0][pos] = x
                    pos += 1
                X_1_mer_one_hot_vectors.append(tmp_1_mer_one_hot_vector)
                Y_train.append(1)
                start = end
                end = start + L
            if remainder_chunk:
                chunk = post_protein[start:start+remainder_chunk]
                tmp_1_mer_one_hot_vector = np.zeros((1, L, 20)) # 1-mer one hot vetors, for chunk
                pos = 0
                for j in chunk:
                    x = np.zeros((1, 20))
                    x[0][kmers_index[j]] = 1
                    tmp_1_mer_one_hot_vector[0][pos] = x
                    pos += 1
                X_1_mer_one_hot_vectors.append(tmp_1_mer_one_hot_vector)
                Y_train.append(1)

    X_1_mer_one_hot_vectors = np.vstack(X_1_mer_one_hot_vectors)
    return X_1_mer_one_hot_vectors

def create_2_mer_one_hot_vectors(positive_proteins, negative_proteins):
    L = 349 # we are using 350 as length of a row in the 3d input matrix as 95% capsid and non-capsid proteins have length less than 350. For 2-mers is 349 instead of 350
    X_2_mer_one_hot_vectors = []
    for i in range(len(negative_proteins)):
        neg_protein = negative_proteins[i].seq
        
        if len(neg_protein) <= L+1:
            tmp_2_mer_one_hot_vector = np.zeros((1, L, 400)) # 1-mer one hot vetors, for neg_protein
            for j in range(len(neg_protein)-1):
                x = np.zeros((1, 400))
                x[0][kmers_index[neg_protein[j:j+2]]] = 1
                tmp_2_mer_one_hot_vector[0][j] = x
            X_2_mer_one_hot_vectors.append(tmp_2_mer_one_hot_vector)
            #Y_train.append(0)
        else:
            number_of_chunks = len(neg_protein) // (L+1)
            remainder_chunk = len(neg_protein) % (L+1)
            start = 0
            end = L+1
            for k in range(number_of_chunks):
                chunk = neg_protein[start:end]
                tmp_2_mer_one_hot_vector = np.zeros((1, L, 400)) # 1-mer one hot vetors, for chunk
                pos = 0
                for j in range(len(chunk)-1):
                    x = np.zeros((1, 400))
                    x[0][kmers_index[chunk[j:j+2]]] = 1
                    tmp_2_mer_one_hot_vector[0][pos] = x
                    pos += 1
                X_2_mer_one_hot_vectors.append(tmp_2_mer_one_hot_vector)
                #Y_train.append(0)
                start = end
                end = start + L+1
            if remainder_chunk:
                chunk = neg_protein[start:start+remainder_chunk]
                tmp_2_mer_one_hot_vector = np.zeros((1, L, 400)) # 1-mer one hot vetors, for chunk
                pos = 0
                for j in range(len(chunk)-1):
                    x = np.zeros((1, 400))
                    x[0][kmers_index[chunk[j:j+2]]] = 1
                    tmp_2_mer_one_hot_vector[0][pos] = x
                    pos += 1
                X_2_mer_one_hot_vectors.append(tmp_2_mer_one_hot_vector)
                #Y_train.append(0)

    for i in range(len(positive_proteins)):
        post_protein = positive_proteins[i].seq
        
        if len(post_protein) <= L+1:
            tmp_2_mer_one_hot_vector = np.zeros((1, L, 400)) # 1-mer one hot vetors, for neg_protein
            for j in range(len(post_protein)-1):
                x = np.zeros((1, 400))
                x[0][kmers_index[post_protein[j:j+2]]] = 1
                tmp_2_mer_one_hot_vector[0][j] = x
            X_2_mer_one_hot_vectors.append(tmp_2_mer_one_hot_vector)
            #Y_train.append(1)
        else:
            number_of_chunks = len(post_protein) // (L+1)
            remainder_chunk = len(post_protein) % (L+1)
            start = 0
            end = L+1
            for k in range(number_of_chunks):
                chunk = post_protein[start:end]
                tmp_2_mer_one_hot_vector = np.zeros((1, L, 400)) # 1-mer one hot vetors, for chunk
                pos = 0
                for j in range(len(chunk)-1):
                    x = np.zeros((1, 400))
                    x[0][kmers_index[chunk[j:j+2]]] = 1
                    tmp_2_mer_one_hot_vector[0][pos] = x
                    pos += 1
                X_2_mer_one_hot_vectors.append(tmp_2_mer_one_hot_vector)
                #Y_train.append(1)
                start = end
                end = start + L+1
            if remainder_chunk:
                chunk = post_protein[start:start+remainder_chunk]
                tmp_2_mer_one_hot_vector = np.zeros((1, L, 400)) # 1-mer one hot vetors, for chunk
                pos = 0
                for j in range(len(chunk)-1):
                    x = np.zeros((1, 400))
                    x[0][kmers_index[chunk[j:j+2]]] = 1
                    tmp_2_mer_one_hot_vector[0][pos] = x
                    pos += 1
                X_2_mer_one_hot_vectors.append(tmp_2_mer_one_hot_vector)
                #Y_train.append(1)

    X_2_mer_one_hot_vectors = np.vstack(X_2_mer_one_hot_vectors)
    return X_2_mer_one_hot_vectors


def create_3_mer_one_hot_vectors(positive_proteins, negative_proteins):
    L = 348 # we are using 350 as length of a row in the 3d input matrix as 95% capsid and non-capsid proteins have length less than 350. For 3-mers is 348 instead of 350
    X_3_mer_one_hot_vectors = []
    for i in range(len(negative_proteins)):
        neg_protein = negative_proteins[i].seq
        
        if len(neg_protein) <= L+2:
            tmp_3_mer_one_hot_vector = np.zeros((1, L, 13)) # 1-mer one hot vetors, for neg_protein
            for j in range(len(neg_protein)-2):
                #x = np.zeros((1, 13))
                x = make_binary_for_3_mers(kmers_index[neg_protein[j:j+3]]+1)
                #x[0][kmers_index[neg_protein[j:j+3]]] = 1
                tmp_3_mer_one_hot_vector[0][j] = x
            X_3_mer_one_hot_vectors.append(tmp_3_mer_one_hot_vector)
            #Y_train.append(0)
        else:
            number_of_chunks = len(neg_protein) // (L+2)
            remainder_chunk = len(neg_protein) % (L+2)
            start = 0
            end = L+2
            for k in range(number_of_chunks):
                chunk = neg_protein[start:end]
                tmp_3_mer_one_hot_vector = np.zeros((1, L, 13)) # 1-mer one hot vetors, for chunk
                pos = 0
                for j in range(len(chunk)-2):
                    #x = np.zeros((1, 400))
                    x = make_binary_for_3_mers(kmers_index[chunk[j:j+3]]+1)
                    #x[0][kmers_index[chunk[j:j+2]]] = 1
                    tmp_3_mer_one_hot_vector[0][pos] = x
                    pos += 1
                X_3_mer_one_hot_vectors.append(tmp_3_mer_one_hot_vector)
                #Y_train.append(0)
                start = end
                end = start + L+2
            if remainder_chunk:
                chunk = neg_protein[start:start+remainder_chunk]
                tmp_3_mer_one_hot_vector = np.zeros((1, L, 13)) # 1-mer one hot vetors, for chunk
                pos = 0
                for j in range(len(chunk)-2):
                    #x = np.zeros((1, 400))
                    x = make_binary_for_3_mers(kmers_index[chunk[j:j+3]] + 1)
                    #x[0][kmers_index[chunk[j:j+2]]] = 1
                    tmp_3_mer_one_hot_vector[0][pos] = x
                    pos += 1
                X_3_mer_one_hot_vectors.append(tmp_3_mer_one_hot_vector)
                #Y_train.append(0)

    for i in range(len(positive_proteins)):
        post_protein = positive_proteins[i].seq
        
        if len(post_protein) <= L+2:
            tmp_3_mer_one_hot_vector = np.zeros((1, L, 13)) # 1-mer one hot vetors, for neg_protein
            for j in range(len(post_protein)-2):
                #x = np.zeros((1, 400))
                x = make_binary_for_3_mers(kmers_index[post_protein[j:j+3]])
                #x[0][kmers_index[post_protein[j:j+2]]] = 1
                tmp_3_mer_one_hot_vector[0][j] = x
            X_3_mer_one_hot_vectors.append(tmp_3_mer_one_hot_vector)
            #Y_train.append(1)
        else:
            number_of_chunks = len(post_protein) // (L+2)
            remainder_chunk = len(post_protein) % (L+2)
            start = 0
            end = L+2
            for k in range(number_of_chunks):
                chunk = post_protein[start:end]
                tmp_3_mer_one_hot_vector = np.zeros((1, L, 13)) # 1-mer one hot vetors, for chunk
                pos = 0
                for j in range(len(chunk)-2):
                    #x = np.zeros((1, 400))
                    x = make_binary_for_3_mers(kmers_index[chunk[j:j+3]]+1)
                    #x[0][kmers_index[chunk[j:j+2]]] = 1
                    tmp_3_mer_one_hot_vector[0][pos] = x
                    pos += 1
                X_3_mer_one_hot_vectors.append(tmp_3_mer_one_hot_vector)
                #Y_train.append(1)
                start = end
                end = start + L+2
            if remainder_chunk:
                chunk = post_protein[start:start+remainder_chunk]
                tmp_3_mer_one_hot_vector = np.zeros((1, L, 13)) # 1-mer one hot vetors, for chunk
                pos = 0
                for j in range(len(chunk)-2):
                    #x = np.zeros((1, 400))
                    x = make_binary_for_3_mers(kmers_index[chunk[j:j+3]]+1)
                    #x[0][kmers_index[chunk[j:j+2]]] = 1
                    tmp_3_mer_one_hot_vector[0][pos] = x
                    pos += 1
                X_3_mer_one_hot_vectors.append(tmp_3_mer_one_hot_vector)
                #Y_train.append(1)

    X_3_mer_one_hot_vectors = np.vstack(X_3_mer_one_hot_vectors)
    return X_3_mer_one_hot_vectors


# fit a multi-headed 1D CNN model
def train_model(trainX1, trainX2, trainX3, trainy):
    verbose, epochs, batch_size = 1, 100, 32
    n_amino_acids1, n_one_hot_vectors1 = trainX1.shape[1], trainX1.shape[2]
    n_amino_acids2, n_one_hot_vectors2 = trainX2.shape[1], trainX2.shape[2]
    n_amino_acids3, n_one_hot_vectors3 = trainX3.shape[1], trainX3.shape[2]
    with tf.device('/gpu:0'):
        # channel 1
        inputs1 = Input(shape=(n_amino_acids1, n_one_hot_vectors1))
        conv1_1 = Conv1D(filters=64, kernel_size=3, activation='relu')(inputs1)
        drop1_1 = Dropout(0.5)(conv1_1)
        pool1_1 = MaxPooling1D(pool_size=2)(drop1_1)
        flat1 = Flatten()(pool1_1)
        # channel 2
        inputs2 = Input(shape=(n_amino_acids2, n_one_hot_vectors2))
        conv2_1 = Conv1D(filters=64, kernel_size=5, activation='relu')(inputs2)
        drop2_1 = Dropout(0.5)(conv2_1)
        pool2_1 = MaxPooling1D(pool_size=2)(drop2_1)
        flat2 = Flatten()(pool2_1)
        # channel 3
        inputs3 = Input(shape=(n_amino_acids3, n_one_hot_vectors3))
        conv3_1 = Conv1D(filters=64, kernel_size=7, activation='relu')(inputs3)
        drop3_1 = Dropout(0.5)(conv3_1)
        pool3_1 = MaxPooling1D(pool_size=2)(drop3_1)
        flat3 = Flatten()(pool3_1)
        # merge
        merged = concatenate([flat1, flat2, flat3])
        drop_after_merging = Dropout(0.5)(merged)
        # interpretation
        dense1 = Dense(100, activation='relu')(drop_after_merging)
        dense2 = Dense(50, activation='relu')(dense1)
        dense3 = Dense(25, activation='relu')(dense2)
        outputs = Dense(1, activation='sigmoid')(dense3)

    model = Model(inputs=[inputs1, inputs2, inputs3], outputs=outputs)
	# save a plot of the model
	#plot_model(model, show_shapes=True, to_file='multichannel.png')
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
	# fit network
    model.fit([trainX1, trainX2, trainX3], trainy, epochs=epochs, batch_size=batch_size, verbose=verbose)
    model.save('capsid_3channels.h5')

non_structural_proteins = list(SeqIO.parse("negative.fasta", "fasta"))
capsid_proteins = list(SeqIO.parse("positive.fasta", "fasta"))
Y_train = []

'''
x_axis = [100, 150, 200, 250, 300, 350, 400]
y_axis = []
for i in x_axis:
    c = find_percentage(i)
    y_axis.append(c)

print(y_axis)
'''

# load the dictionary of kmers' indices from a json file
with open('index_of_kmers.json') as index_file:
    kmers_index = json.load(index_file) # dictionary to hold the index of a k-mer for one hot encoding



X_1_mers = create_1_mer_one_hot_vectors(capsid_proteins, non_structural_proteins)
#print(X_1_mers.shape[1], X_1_mers.shape[2], sep=' ')
X_2_mers = create_2_mer_one_hot_vectors(capsid_proteins, non_structural_proteins)
#print(X_2_mers.shape[1], X_2_mers.shape[2], sep=' ')
X_3_mers = create_3_mer_one_hot_vectors(capsid_proteins, non_structural_proteins)
#print(X_3_mers.shape[1], X_3_mers.shape[2], sep=' ')
Y_train = np.array(Y_train)
#print(Y_train.shape)

train_model(X_1_mers, X_2_mers, X_3_mers ,Y_train)

'''
L = 5000
demo_neg_seq = negative_proteins[0].seq
demo_3d_array = np.zeros((1, L, 20)) # 1-mer one hot vetors
for i in range(len(demo_neg_seq)):
    x = np.zeros((1, 20))
    x[0][kmers_index[demo_neg_seq[i]]] = 1
    demo_3d_array[0][i] = x

print(demo_neg_seq)
print(demo_3d_array[0][1])
print(demo_3d_array[0][2])
print(demo_3d_array[0][2123])
'''
