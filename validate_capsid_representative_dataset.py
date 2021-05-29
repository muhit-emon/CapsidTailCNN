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

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


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
            #Y_train.append(0)
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
                #Y_train.append(0)
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
                #Y_train.append(0)

    for i in range(len(positive_proteins)):
        post_protein = positive_proteins[i].seq
        
        if len(post_protein) <= L:
            tmp_1_mer_one_hot_vector = np.zeros((1, L, 20)) # 1-mer one hot vetors, for neg_protein
            for j in range(len(post_protein)):
                x = np.zeros((1, 20))
                x[0][kmers_index[post_protein[j]]] = 1
                tmp_1_mer_one_hot_vector[0][j] = x
            X_1_mer_one_hot_vectors.append(tmp_1_mer_one_hot_vector)
            #Y_train.append(1)
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
                #Y_train.append(1)
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
                #Y_train.append(1)

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

non_structural_proteins = list(SeqIO.parse("../capsid_testing/representative/negative.fasta", "fasta"))
capsid_proteins = list(SeqIO.parse("../capsid_testing/representative/positive.fasta", "fasta"))

Y_test_true = []

# pre-processing of validation data. The following code snippet finds the chunks of each validation protein sequence
spans_of_chunks = []
L = 350 # each protein will be either be padded with zeors or be split into chunks of length 350
strt = 0
end = 0
for i in range(len(non_structural_proteins)):
    l = len(non_structural_proteins[i].seq)
    if l <= L:
        end = strt
        spans_of_chunks.append((strt, end))
    else:
        n_of_chunks = l // L
        if l % L:
            n_of_chunks += 1
        end = strt + n_of_chunks - 1
        spans_of_chunks.append((strt, end))
    strt = end + 1
    Y_test_true.append(0)

for i in range(len(capsid_proteins)):
    l = len(capsid_proteins[i].seq)
    if l <= L:
        end = strt
        spans_of_chunks.append((strt, end))
    else:
        n_of_chunks = l // L
        if l % L:
            n_of_chunks += 1
        end = strt + n_of_chunks - 1
        spans_of_chunks.append((strt, end))
    strt = end + 1
    Y_test_true.append(1)

Y_test_true = np.array(Y_test_true)

with open('index_of_kmers.json') as index_file:
    kmers_index = json.load(index_file)

X_1_mers = create_1_mer_one_hot_vectors(capsid_proteins, non_structural_proteins)
#print(X_1_mers.shape)
X_2_mers = create_2_mer_one_hot_vectors(capsid_proteins, non_structural_proteins)
#print(X_2_mers.shape)
X_3_mers = create_3_mer_one_hot_vectors(capsid_proteins, non_structural_proteins)

# evaluate saved model on validation dataset with 960 capsid(p) proteins and 1576 non-structural(n) proteins
model = load_model('capsid_3channels.h5')
Y_prediction = model.predict([X_1_mers, X_2_mers, X_3_mers])

Y_prediction_processed = []
for i in range(len(spans_of_chunks)):
    s, e = spans_of_chunks[i]
    mx = -1000
    for j in range(s, e+1):
        mx = max(mx, Y_prediction[j][0])
    Y_prediction_processed.append(mx)

th = 0.5

Y_pred = [] # this list contains either 1 or 0 for depending on score threshold
for i in range(len(Y_prediction_processed)):
    if Y_prediction_processed[i] >= th:
        Y_pred.append(1)
    else:
        Y_pred.append(0)


# confusion matrix for score threshold 'th' which is a variable
tn, fp, fn, tp = confusion_matrix(Y_test_true, Y_pred).ravel()
print(tn, fp, fn, tp, sep=' ')

p = tp / (tp + fp)
r = tp / (tp + fn)
f1 = (2*r*p)/ (r+p)
acc = (tp + tn) / (tn + fp + fn + tp)

print(f"Precision: {p}")
print(f"Recall: {r}")
print(f"F1-score: {f1}")
print(f"Accuracy: {acc}")
