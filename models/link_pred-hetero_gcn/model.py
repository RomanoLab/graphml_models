#!/usr/bin/python3

######################################################################################
# README!
# Link prediction model adapted from various sources online and joe's code.
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

DEVICE = 'cpu'

######################################################################################
# GET NODE REPRESENTATIONS USING RCGN LAYER
######################################################################################
class HeteroRGCNLayer(nn.Module):
    '''
    RCGN LAYER TO CALCULATE NODE REPRESENTATIONS.
    '''
    def __init__(self, in_size, out_size, etypes):
        super().__init__()

        if isinstance(in_size, dict):
            self.weight = nn.ModuleDict({
                name: nn.Linear(in_size[name], out_size).to(DEVICE) for name in etypes
            })
        else:
            self.weight = nn.ModuleDict({
                name: nn.Linear(in_size, out_size).to(DEVICE) for name in etypes
            })

    def forward(self, positive_graph, feat_dict):
        '''
        Calculate node representation and add as feature to nodes.

        Parameters
        ----------
        positive_graph : graph you want to calculate node representations for
        feat_dict : dictionary of node features for each node type

        Outputs
        ----------
        h : dictionary of representation for each node
        '''
         
        funcs = {}

        for srctype, etype, dsttype in positive_graph.canonical_etypes: 
            Wh = self.weight[etype](feat_dict[srctype]).to(DEVICE) # get weight of edge for each source node type 
            positive_graph.nodes[srctype].data['Wh_%s' % etype] = Wh # set a feature ‘Wh_etype’ for all nodes of the source node type
            funcs[etype] = (fn.copy_u('Wh_%s' % etype, 'm'), fn.mean('m', 'h')) # still a bit confused ??????

        positive_graph.multi_update_all(funcs, 'sum') # send messages along all the edges, update the node features of all the nodes
        h = { ntype : positive_graph.nodes[ntype].data['h'] for ntype in positive_graph.ntypes }

        return h

######################################################################################
# SCORE PREDICTOR 
######################################################################################
class HeteroDotProductPredictor(nn.Module):
    '''
    MAKE SCORE PREDICTION USING DOT PRODUCT.
    '''
    def forward(self, edge_subgraph, h, etype):
        '''
        Predict edge score and add as feature to edges.

        Parameters
        ----------
        edge_subgraph : graph you want to calculate edge scores for
        h : node representation from RGCN
        etype : edge type you want to be making score prediction for

        Outputs
        ----------
        edges_scores : scores for each edge
        '''
        
        # we might want to do for all edges
        with edge_subgraph.local_scope():
            edge_subgraph.ndata['h'] = h # get the node representation
            edge_subgraph.apply_edges(fn.u_dot_v('h', 'h', 'score'), etype = etype) # update features of specified edge with provide function 
            edges_scores = edge_subgraph.edges[etype].data['score']

            return edges_scores

######################################################################################
# LINK PREDICTION
######################################################################################
class Model(nn.Module):
    '''
    THE LINK PREDICTION MODEL.
    '''
    def __init__(self, in_size_dict, hidden_size, out_size, rel_names):
        super().__init__()

        self.sage = HeteroRGCNLayer(in_size_dict, out_size, rel_names)
        self.pred = HeteroDotProductPredictor()

    def forward(self, positive_graph, negative_graph, x, etype):
        '''
        Get scores for positive and negative graph.

        Parameters
        ----------
        positive_graph : positive graph (the graph with the edges actually in our graph)
        negative_graph : negative graph (the graph with the edges that we negative sampled)
        x : node features
        etype : edge type we want to make score predictions for

        Outputs
        ----------
        pos_score : scores for each edge in positive graph (each edge of our edge type of course)
        neg_score : scores for each edge in negative graph
        '''

        h = self.sage(positive_graph, x) # node representations 

        pos_score = self.pred(positive_graph, h, etype)
        neg_score = self.pred(negative_graph, h, etype)

        return pos_score, neg_score
