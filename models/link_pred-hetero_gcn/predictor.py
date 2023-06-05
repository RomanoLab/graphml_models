#!/usr/bin/python3

######################################################################################
# README!
# Code to get predictions from trained link prediction model.
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
def predict(pos_scores, neg_scores, negative_test_graph, lp_etype):
    '''
    Predict new edges.

    Parameters
    ----------
    pos_scores : positive scores
    neg_scores : negative scores
    negative_test_graph : negative test graph
    lp_etype : edge we want predictions on 

    Outputs
    ----------
    prediction_df : dataframes of src,dst nodes of edges that are predicted to exist

    '''
    # we choose which neg_scores to keep based on a threshold made from the pos_scores
    # make threshold
    pos_score_mean = torch.mean(pos_scores)
    pos_score_sd = torch.std(neg_scores)

    threshold = pos_score_mean - pos_score_sd

    # keep only the negative scores above set threshold
    neg_idx = neg_scores >= threshold

    # we find the source and destination nodes for each edge in the negative graph 
    src = negative_test_graph.edges(etype = lp_etype)[0]
    dst = negative_test_graph.edges(etype = lp_etype)[1]

    # need to fix the shapes
    src_1D = torch.empty(src.size(0), 1)
    dst_1D = torch.empty(dst.size(0), 1)

    src_1D[:,0] = src
    dst_1D[:,0] = dst

    # source and destination nodes of negative edges that are predicted to 'exist'
    src_nodes_pred = src_1D[neg_idx]
    dst_nodes_pred = dst_1D[neg_idx]

    # save as dataframe
    data = {'src': src_nodes_pred,
        'dst': dst_nodes_pred, 
        'pred_score': neg_scores[neg_idx]}
  
    prediction_df = pd.DataFrame(data)
    prediction_df.to_csv(index=False)

    return prediction_df

if __name__=="__main__":
    ######################################################################################
    # INITIALIZATION OF ARGUMENTS
    ######################################################################################
    parser = argparse.ArgumentParser(description = "Make predictions based on trained link prediction model.")

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

    # not sure what this does
    parser.set_defaults(validation = True) # not sure what this does

    args = parser.parse_args()

    ######################################################################################
    # MAKE PREDICTIONS 
    ######################################################################################
    print("Making predictions...")
    # set edge type for link prediction
    lp_etype = args.edge_type

    # get scores to make predictions 
    pos_scores = torch.load(args.pos_file)
    neg_scores = torch.load(args.neg_file)

    # get negative_test_graph
    negative_test_graph = dgl.load_graphs(args.ngraph_file)[0][0]

    # predictions
    prediction_df = predict(pos_scores, neg_scores, negative_test_graph, lp_etype)

    # save 
    prediction_df.to_csv(index=False)

    # print a handful of the predicitons
    df_first_10 = prediction_df.head(10)
    print(df_first_10)