 
import networkx as nx
import random
import copy
import pandas as pd
import ast
from ast import literal_eval
from scipy import stats
import numpy as np
 
import numpy as np
import networkx as nx
import pandas as pd
import math
# ---------------------------------------------------------------------
from scipy import stats
from keras.layers import Conv2D, BatchNormalization, ReLU,AveragePooling2D, MaxPooling2D, Input, Dense, Flatten,concatenate
from keras.models import Model
import matplotlib.pyplot as plt
from scipy import stats
import os
import tensorflow as tf
import random

import keras
import tensorflow


import networkx as nx
import pandas as pd
# import matplotlib.pyplot as plt
# from random import uniform, seed
# import numpy as np
import os

import time
from operator import itemgetter # for sort list of list
# from math import e
import math


 
def get_degree(G):
    # Create an empty dictionary to store the node degrees
    degree = {}
    # Loop through all nodes in the graph
    for node in G.nodes():
        # Get the degree of the current node by accessing the 'degree' property
        # of the graph object and Add an entry to the dictionary with the node as the key and its degree as the value
        degree[node] = math.log(G.degree[node])
    # Return the dictionary of node degrees
    return degree


def get_probability_ratios(G):
    # Initialize an empty dictionary to store the result
    degree_ratios = {}
    # Loop through the nodes in the graph
    for node in G.nodes():
        # Store the result in the dictionary with the node as the key
        #degree_ratios[node] =(1/G.degree[node])/(1+F_C1_[node])
        degree_ratios[node] =(1/G.degree[node])
    # Return the final dictionary
    return degree_ratios


def compute_node_weights(G):
    weights = {}
    for node in G.nodes():
        node_degree = G.degree(node)
        neighbor_sum = sum([1/G.degree(neighbor) for neighbor in G.neighbors(node)])
        node_weight = neighbor_sum / node_degree
        weights[node] = node_weight
    return weights




def weight_1hop(G):
    dicW={}
    # Iterate over all nodes in the graph
    for node in G.nodes():
        # Get the neighbors of the current node
        neighbors = list(G.neighbors(node))
        # Iterate over the neighbors of the current node
        sumWW=0
        for neighbor in neighbors:
            # Get the neighbors of the current neighbor, excluding the current node
            neighbor_neighbors = list(G.neighbors(neighbor))
            neighbor_neighbors.remove(node)

            # Iterate over the neighbors of the current neighbor
            for neighbor_neighbor in neighbor_neighbors:
                # Check if the current neighbor's neighbor is also a neighbor of the original node
                if neighbor_neighbor in neighbors:
                    # Record the connection between the two neighbors of the original node
                    #print(f"Node {node} connects {neighbor} and {neighbor_neighbor}")
                    sumWW+=(1/G.degree(node))*(1/G.degree(neighbor_neighbor))*(1/G.degree(neighbor))
                    #print(dicW[node])
        dicW[node]=sumWW
    return dicW

def weight_2hop(G,weights):
    dicW={}
    # Iterate over all nodes in the graph
    for node in G.nodes():
        neighbors = list(G.neighbors(node))
        sumWW=0
        for neighbor in neighbors:
            sumWW+=weights[neighbor]-((1/G.degree(node))*(1/G.degree(neighbor)))
        dicW[node]=sumWW*(1/G.degree(node))
    return dicW





def feature_extrating(G):
    F_d1_r=get_probability_ratios(G)
    F_d2_r=compute_node_weights(G)
    F_d3_r=weight_1hop(G)
    F_d4_r=weight_2hop(G,F_d2_r)


    return  F_d1_r,F_d2_r,F_d3_r ,F_d4_r

def weighted_adjacency_matrix_3channels(G, node_name, L,   F11,F22,F33,F44):
    # Get the neighbors of the given node
    neighbors = G[node_name]

    # Get the top L neighbors based on their degree
    degree = {}
    for node in neighbors:
        degree[node] = G.degree(node)

    sorted_degree = sorted(degree.items(), key=lambda x: x[1], reverse=True)
    top_L_neighbors = [node[0] for node in sorted_degree[:L]]
    top_L_neighbors = [node_name] + top_L_neighbors


    matrix_F2 = np.zeros((4,L+1), dtype=float)

    for i, node_1 in enumerate(top_L_neighbors):

        matrix_F2[0,i]=F11[node_1]
        matrix_F2[1,i]=F22[node_1]
        matrix_F2[2,i]=F33[node_1]
        matrix_F2[3,i]=F44[node_1]


    matrix_F2 = np.array(matrix_F2)  # your weight vectors

    return matrix_F2
    #return np.dstack((matrix_F1, matrix_F2,matrix_F3))



def get_data_to_model_3channels(G,L):

    P_F1,P_F2,P_F3,P_F4=feature_extrating(G)

    x_train2 = []


    for node in G:
        matrix_F2=weighted_adjacency_matrix_3channels(G, node, L, P_F1,P_F2,P_F3,P_F4)

        x_train2.append(matrix_F2)


    x_train2 = np.array(x_train2).astype(float)


    return x_train2

def get_labels_to_model(G,labels_G):
    y_train=[]
    for node in G:
        y_train.append(labels_G[str(node)])
    y_train = np.array(y_train).astype(float)
    return y_train





 

# ---------------------------------------------------------------------
 

k=11111
#k=12321

def reset_random_seeds():
    os.environ['PYTHONHASHSEED']=str(k)
    tf.random.set_seed(k)
    np.random.seed(k)
    random.seed(k)
reset_random_seeds()


L=10
 
algoname='NDM'
 
 
def NDM_rank(G, L=10):
  modelname='/content/model-NDM[C4]-100.h5'
  model = keras.models.load_model(modelname)

  x2_train=get_data_to_model_3channels(G,L)

  x2 = np.concatenate(( x2_train,))
  print(x2.shape)

  data_predictions = model.predict([x2])
  nodes = list(G.nodes())
  seed = [i for i,j in sorted(dict(zip(nodes,data_predictions)).items(),key=lambda x:x[1],reverse=True)]

  return seed


 



 