#!/usr/bin/python3

######################################################################################
# README!
# Link prediction adapted from various sources online and joe's code.
# This link prediction is adapted to work on heterogenous graphs and make predictions on 
# one edge type.
# This code can be adapted to work with different heterogenous graph but we have:
# - 4 node types
# - 14 edge types
# - predictions are made on one edge type: CHEMICALASSOCIATESWITHDISEASE
######################################################################################

import argparse

import dgl
from dgl import save_graphs
import dgl.function as fn
import dgl.nn as dglnn
from os import path

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

from model import Model

DEVICE = 'cpu'

######################################################################################
# PROCESS GRAPH 
######################################################################################
def preprocess_edges(graph):
    '''
    Preprocess graph for link prediction.

    Parameters
    ----------
    graph : dgl graph you want to preprocess

    Outputs
    ----------
    node_features : dictionary with each node type and a tensor of ones of length # of number of nodes within that node type 
    node_sizes : dictionary of each node and its tensor column length
    edge_input_sizes : dictionary of each edges with tensor column length of the source node corresponding to the edge
    ** this is done so that if we have multiple features for each node we can take it into account
    '''
    # when we will add the maacs as features:
    # chem = pd.read_csv("/Users/cfparis/Desktop/Romano_Rotation/graph_maker/data/chemicals.csv") 
    # maccs = torch.tensor([[int(x) for x in xx] for xx in chem.maccs]).float().to(DEVICE)

    node_features = {
        #'chemical': maccs,
        'chemical': torch.ones((graph.number_of_nodes(ntype='chemical'))).unsqueeze(1).to(DEVICE),
        'assay': torch.ones((graph.number_of_nodes(ntype='assay'))).unsqueeze(1).to(DEVICE),
        'gene': torch.ones((graph.number_of_nodes(ntype='gene'))).unsqueeze(1).to(DEVICE),
        'disease': torch.ones((graph.number_of_nodes(ntype='disease'))).unsqueeze(1).to(DEVICE)
    }

    input_type_map = dict([(x[1], x[0]) for x in graph.canonical_etypes]) # each edge type with the source node 
    node_sizes = { k: v.shape[1] for k, v in node_features.items() }
    edge_input_sizes = { k: node_features[v].shape[1] for k, v in input_type_map.items() }

    return node_features, node_sizes, edge_input_sizes

######################################################################################
# MAKE A NEGATIVE GRAPH
######################################################################################
def construct_negative_graph(graph, k): 
    '''
    Construct a negative graph for negative sampling in edge prediction.

    Parameters
    ----------
    G : dgl graph you want to make negative graph for
    k : number of negative examples to retrieve

    Outputs
    ----------
    m_neg_graph : negative graph 
    '''

    m_neg_graph = dgl.heterograph(
        {etype: (graph.edges(etype = etype)[0].repeat_interleave(k), torch.randint(0, graph.num_nodes(etype[2]), (len(graph.edges(etype = etype)[0]) * k,))) for etype in graph.canonical_etypes}
    )

    return m_neg_graph

######################################################################################
# COMPUTE LOSS
######################################################################################
def compute_loss(pos_score, neg_score):
    '''
     Get scores for positive and negative graph.

    Parameters
    ----------
    pos_score : scores for each edge in positive graph (each edge of our edge type of course)
    neg_score : scores for each edge in negative graph

    Outputs
    ----------
    hinge_loss : hinge loss that compares score between nodes connected by an edge in positive graph against score between nodes connected by an edge in negative graph
    '''
   
    n = pos_score.shape[0]
    hinge_loss = (neg_score.view(n, -1) - pos_score.view(n, -1) + 1).clamp(min=0).mean()

    return hinge_loss

######################################################################################
# TRAINING OF MODEL
######################################################################################
def training(device, graph, model, train_eid_dict, reverse_edges, c_lp_etype):
    '''
    Train link prediction model.

    Parameters
    ----------
    device : device we want to use for training
    G : graph we want to train on
    model : our model we will use for training
    train_eid_dict : dictionary of edge types and edge IDs
    c_lp_etype : canonical edge type we are making prediction on

    Outputs
    ----------
    all_loss : our model loss
    '''

    ################# Sampler and dataloader #########################################
    ## for each source node of each edge in our graph we pick k (in our case 1) destination nodes uniformly 
    # number of layers and how many messages you are taking from each layer 
    # could also use: sampler = NeighborSampler([15, 10, 5])
    sampler = dgl.dataloading.MultiLayerFullNeighborSampler(2) 
    
    # we want to turn our sampler into one that can be used for edge prediction instead of node classification
    sampler = dgl.dataloading.as_edge_prediction_sampler(
        sampler, 
        exclude = "reverse_types", # exclude edges in the current minibatch but also their reverse edges according to the ID mapping in the argument reverse_eids.
        reverse_etypes = reverse_edges, # reverse edges
        negative_sampler = dgl.dataloading.negative_sampler.Uniform(2) # our negative sampler ***a parameter we can probably tweak here***
        ) 

    ## with a negative sampler, dataloader will generate 3 items per minibatch:
    # a positive graph containing all the edges sampled in the minibatch.
    # a negative graph containing all the non-existent edges generated by the negative sampler.
    # a list of message flow graphs (MFGs) generated by the neighborhood sampler.

    dataloader = dgl.dataloading.DataLoader(
        graph, # graph
        train_eid_dict, # dict of edge type and edge ID tensors
        sampler, # sampler
        device = device, # device we want to use 
        batch_size = 512, # number of indices in each batch ***a parameter we can probably tweak here***
        shuffle = True, # randomly shuffle the indices at each epoch
        drop_last = True, # don't drop the last incomplete batch
        num_workers = 0 # not completely sure what this does i think it is a computer thing
        )
    
    ################# Train model #########################################
    opt = torch.optim.Adam(model.parameters())  # optimization algorithm we want to use

    print('...Running epochs')

    # where we will be storing our loss
    all_loss = []

    # iterates over data loader and feeds in the small graphs + input features to the model defined above
    for epoch in range(1):
        model.train()
        total_loss = 0

        for it, (input_nodes, positive_graph, negative_graph, blocks) in enumerate(dataloader):
            # blocks of data
            blocks = [b.to(torch.device('cpu')) for b in blocks]

            # the positive graph and the negative graph
            positive_graph = positive_graph.to(torch.device('cpu'))
            negative_graph = negative_graph.to(torch.device('cpu'))

            # input features
            input_features = blocks[0].srcdata['features']
 
            # preprocess the graph
            sm_node_features, sm_node_sizes, sm_edge_input_sizes = preprocess_edges(positive_graph)

            # get scores from model
            pos_score, neg_score = model(positive_graph, negative_graph, sm_node_features, c_lp_etype)

            # compute loss
            loss = compute_loss(pos_score, neg_score)
            opt.zero_grad()
            loss.backward()
            opt.step()

            # get total loss
            total_loss += loss.item()

            # keep track of loss
            all_loss.append(total_loss / (it + 1))

            if (it + 1) == 1000:
                break

        print("Epoch {:05d} | Loss {:.4f}".format(epoch, total_loss / (it + 1)))

    return all_loss

if __name__=="__main__":
    ######################################################################################
    # INITIALIZATION OF ARGUMENTS
    ######################################################################################
    parser = argparse.ArgumentParser(description = "Train a heterogeneous RGCN on a link prediction task.")

    # give location of graph file
    parser.add_argument("--graph-file", type=str, default = "/Users/cfparis/Desktop/romano_lab/graphml_models/models/link_pred-hetero_gcn/data/graph.bin",
                        help = "File location where the DGL heterograph is stored.")

    # give location of edge list
    parser.add_argument("--edge-file", type = str, default = "/Users/cfparis/Desktop/romano_lab/graphml_models/models/link_pred-hetero_gcn/data/edge_dict.pkl",
                        help = "File location where the edge list is stored.")
    
    # give edge type you want to make predictions on 
    parser.add_argument("--edge-type", type = str, default = "chemicalassociateswithdisease",
                        help = "Edge type to make link predictions on.")
    
    parser.add_argument("--cedge-type", type = str, default = ('chemical', 'chemicalassociateswithdisease', 'disease'),
                        help = "Canonical edge type to make link predictions on.")

    # not sure what this does
    parser.set_defaults(validation = True) # not sure what this does

    args = parser.parse_args()

    ######################################################################################
    # INITIALIZATION OF GRAPH AND MODEL
    ######################################################################################
    # set edge type for link prediction
    lp_etype = args.edge_type
    c_lp_etype = args.cedge_type

    # choose mode to train in
    print("Training in cpu mode.")
    device = torch.device('cpu')
    
    print('Loading in graph...')
    G = dgl.load_graphs(args.graph_file)[0][0]
    G = G.to("cpu")


    print('Getting reverse IDs...')
    with open(args.edge_file, 'rb') as f:
        reverse_edges = pickle.load(f)
    
    print('Preprocessing graph...')
    node_features, node_sizes, edge_input_sizes = preprocess_edges(G)

    # dictionary of edge types and edge IDs
    train_eid_dict = {etype: G.edges(etype = etype, form = 'eid') for etype in G.canonical_etypes}

    print('Making model...')
    model = Model(edge_input_sizes, 20, 5, G.etypes)

    ######################################################################################
    # TRAIN MODEL 
    ######################################################################################
    print("Training...")
    loss_array = training(device, G, model, train_eid_dict, reverse_edges, c_lp_etype)

    print("Making loss graph...")
    loss_plt = sns.lineplot(data = loss_array)
    loss_fig = loss_plt.figure
    loss_fig.savefig('loss_fig.png')

    print('Saving data for prediction...')
    # save negative graph for prediction taks
    negative_test_graph = construct_negative_graph(G, 5)

    output_filename = path.join("data", "ngraph.bin")
    save_graphs(output_filename, negative_test_graph)
    
    # save scores for prediction 
    with torch.no_grad():  
        node_features, node_sizes, edge_input_sizes = preprocess_edges(G)
        pos_score, neg_score = model(G, negative_test_graph, node_features, c_lp_etype)
    
        # save tensor 
        torch.save(pos_score, 'pos_score.pt')
        torch.save(neg_score, 'neg_score.pt')