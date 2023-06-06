#!/usr/bin/python3

######################################################################################
# README!
# Code to get visualize predictions from trained link prediction model.
######################################################################################
import argparse

import dgl
import dgl.function as fn
import dgl.nn as dglnn

import pickle
import numpy as np
import sklearn.metrics
import seaborn as sns
import torch
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

import torch.nn as nn
import torch.nn.functional as F

from dgl.dataloading import (
    as_edge_prediction_sampler,
    DataLoader,
    MultiLayerFullNeighborSampler,
    negative_sampler,
    NeighborSampler,
)

DEVICE = 'cpu'

######################################################################################
# PREDICTION USING MODEL
######################################################################################
def visualize(pos_scores, neg_scores, positive_test_graph, negative_test_graph, lp_etype):
    '''
    Predict new edges.

    Parameters
    ----------
    pos_scores : positive scores
    neg_scores : negative scores
    positive_test_graph : positive test graph
    negative_test_graph : negative test graph
    lp_etype : edge we want predictions on 

    Outputs
    ----------
    G : networkx graph
    '''
    # we find the source and destination nodes for each edge in negative and positive graph
    src_neg = negative_test_graph.edges(etype = lp_etype)[0]
    dst_neg = negative_test_graph.edges(etype = lp_etype)[1]

    src_pos = positive_test_graph.edges(etype = lp_etype)[0]
    dst_pos = positive_test_graph.edges(etype = lp_etype)[1]

    # need to fix the shapes
    src_1D_neg = torch.empty(src_neg.size(0), 1)
    dst_1D_neg = torch.empty(dst_neg.size(0), 1)

    src_1D_neg[:,0] = src_neg
    dst_1D_neg[:,0] = dst_neg

    src_1D_pos = torch.empty(src_pos.size(0), 1)
    dst_1D_pos = torch.empty(dst_pos.size(0), 1)

    src_1D_pos[:,0] = src_pos
    dst_1D_pos[:,0] = dst_pos

    # make negative edge tuples
    neg_edge_tuples = []
    src_ID_neg = src_1D_neg.squeeze().tolist()
    dest_ID_neg = dst_1D_neg.squeeze().tolist()
    neg_edge_weight_list = neg_scores.squeeze().tolist()

    for index, node_id in enumerate(src_ID_neg):
        new_element = (
            node_id,
            dest_ID_neg[index],
            { 'weight': neg_edge_weight_list[index]}
        )

        neg_edge_tuples.append(new_element)

    # threshold edges
    neg_edge_tuples_threshold = [ x for x in neg_edge_tuples if (x[2]['weight'] >= 3.0) ]

    # make positive edge tuples
    pos_edge_tuples = []
    src_ID_pos = src_1D_pos.squeeze().tolist()
    dest_ID_pos = dst_1D_pos.squeeze().tolist()
    pos_edge_weight_list = pos_scores.squeeze().tolist()

    for index_p, node_id_p in enumerate(src_ID_pos):
        new_element_p = (
            node_id_p,
            dest_ID_pos[index_p],
            { 'weight': pos_edge_weight_list[index_p]}
        )

        pos_edge_tuples.append(new_element_p)

    # threshold edges
    #neg_edge_tuples = [ x for x in neg_edge_tuples if (x[2]['weight'] >= 6) ]
    #pos_edge_tuples = [ x for x in pos_edge_tuples if (x[2]['weight'] >= 6) ]

    pos_edge_tuples = pos_edge_tuples[30:100]
    neg_edge_tuples = neg_edge_tuples[30:100]

    # add edges to networkx graph with weights and edge type as attributes
    G = nx.Graph()
    G.add_edges_from(pos_edge_tuples, color = 'green')
    G.add_edges_from(neg_edge_tuples, color = 'red')

    # add node attributes (chemical or disease)
    for index, node_id in enumerate(neg_edge_tuples):
        G.add_node(node_id[0], ncolor = 'purple')
        G.add_node(node_id[1], ncolor = 'blue')

    for index, node_id in enumerate(pos_edge_tuples):
        G.add_node(node_id[0], ncolor = 'purple')
        G.add_node(node_id[1], ncolor = 'blue')

    return G

if __name__=="__main__":
    ######################################################################################
    # INITIALIZATION OF ARGUMENTS
    ######################################################################################
    parser = argparse.ArgumentParser(description = "Make predictions based on trained link prediction model.")

    # give location of positive graph file
    parser.add_argument("--graph-file", type=str, default = "/Users/cfparis/Desktop/romano_lab/graphml_models/models/link_pred-hetero_gcn/data/graph.bin",
                        help = "File location where the DGL positive heterograph is stored.")

    # give location of negative graph file
    parser.add_argument("--ngraph-file", type=str, default = "/Users/cfparis/Desktop/romano_lab/graphml_models/models/link_pred-hetero_gcn/data/ngraph.bin",
                        help = "File location where the DGL negative heterograph is stored.")
    
    # give location of pos_score file
    parser.add_argument("--pos-file", type = str, default = "/Users/cfparis/Desktop/romano_lab/graphml_models/models/link_pred-hetero_gcn/pos_score.pt",
                        help = "File location where the edge list is stored.")
    
    # give location of neg_score file
    parser.add_argument("--neg-file", type = str, default = "/Users/cfparis/Desktop/romano_lab/graphml_models/models/link_pred-hetero_gcn/neg_score.pt",
                        help = "File location where the edge list is stored.")
    
    # give edge type you want to make predictions on 
    parser.add_argument("--edge-type", type = str, default = "chemicalassociateswithdisease",
                        help = "Edge type to make link predictions on.")
    
    # give location where to save csv
    parser.add_argument("--file-path", type = str, default = "/Users/cfparis/Desktop/romano_lab/graphml_models/models/link_pred-hetero_gcn/data/",
                        help = "File path where to save the csv files.")

    # not sure what this does
    parser.set_defaults(validation = True) # not sure what this does

    args = parser.parse_args()

    ######################################################################################
    # VISUALIZATION 
    ######################################################################################
    print("Making visualization...")
    # set edge type for link prediction
    lp_etype = args.edge_type

    # get scores to make predictions 
    pos_scores = torch.load(args.pos_file)
    neg_scores = torch.load(args.neg_file)

    # get positive_test_graph
    positive_test_graph = dgl.load_graphs(args.graph_file)[0][0]

    # get negative_test_graph
    negative_test_graph = dgl.load_graphs(args.ngraph_file)[0][0]

    # networkx graph
    G = visualize(pos_scores, neg_scores, positive_test_graph, negative_test_graph, lp_etype)

    # save scores
    scores = nx.get_edge_attributes(G, 'weight').values()
    scores_df = pd.DataFrame(scores)
    scores_df.to_csv(args.file_path + 'graph_scores.csv', index=False)

    # plot 
    position = nx.spring_layout(G, k = 0.4) # layout of nodes can change k to change node positions
    ecolors = nx.get_edge_attributes(G, 'color').values() # color of edges based on edge type (postive or negative)
    weights = nx.get_edge_attributes(G, 'weight').values() # width of edges based on scores
    ncolors = nx.get_node_attributes(G, 'ncolor').values() # color of nodes based on node type (chemical or disease)

    nx.draw(G, position, node_color = ncolors, edge_color = ecolors, width = list(weights))

    plt.show()
