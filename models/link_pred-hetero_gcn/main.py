## main file to run link prediction 
# using code from Joe Romano's qsar paper

## packages
import argparse
import dgl
import numpy as np
import pandas as pd
import pickle as pkl
import torch
import torch.nn.functional as F

from link_prediction import LinkPredictor, compute_lp_loss

DEVICE='cpu'

## functions
def preprocess_edges(graph):
    chem = pd.read_csv("/Users/cfparis/Desktop/Romano_Rotation/GNN/data/chemicals.csv") 
    maccs = torch.tensor([[int(x) for x in xx] for xx in chem.maccs]).float().to(DEVICE)
    node_features = {
        'chemical': maccs,
        #'chemical': torch.ones((graph.number_of_nodes(ntype='chemical'))).unsqueeze(1).to(DEVICE),
        'assay': torch.ones((graph.number_of_nodes(ntype='assay'))).unsqueeze(1).to(DEVICE),
        'gene': torch.ones((graph.number_of_nodes(ntype='gene'))).unsqueeze(1).to(DEVICE),
        'disease': torch.ones((graph.number_of_nodes(ntype='disease'))).unsqueeze(1).to(DEVICE)
    }
    input_type_map = dict([(x[1], x[0]) for x in graph.canonical_etypes])
    node_sizes = { k: v.shape[1] for k, v in node_features.items() }
    edge_input_sizes = { k: node_features[v].shape[1] for k, v in input_type_map.items() }

    return node_features, node_sizes, edge_input_sizes
    
def construct_negative_graph(graph, k, etype): 
    """Construct a negative graph for negative sampling in edge prediction.
    
    This implementation is designed for heterogeneous graphs - the user specifies
    the edge type on which negative sampling will be performed.

    Parameters
    ----------
    graph : dgl.heterograph.DGLHeterograph
        Graph on which the sampling will be performed.
    k : int
        Number of negative examples to retrieve.
    etype : tuple
        A tuple in the format (subj_node_type, edge_label, obj_node_type) corresponding
        to the edge on which negative sampling will be performed.
    """
    utype, _, vtype = etype
    src, dst = graph.edges(etype=etype)

    neg_src = src.repeat_interleave(k)
    neg_dst = torch.randint(0, graph.num_nodes(vtype), (len(src) * k,))
    
    return dgl.heterograph(
        {etype: (neg_src, neg_dst)},
        num_nodes_dict= { ntype: graph.num_nodes(ntype) for ntype in graph.ntypes }
    )

def link_prediction(args):
    """Predict edges in a heterogeneous graph given a particular edge type.

    For this implementation, the edge type we are predicting is:
    `('chemical', 'chemicalassociateswithdisease', 'disease')`

    There are two approaches for training the network:
    1. Train known edges against a negative sampling of the entire graph, using
       margin loss (or equivalent) to maximize the difference between known edges
       and the background "noise distribution" of randomly sampled edges.
    2. Use a predetermined edge instead as the negative graph. 
       This approach may be more powerful. Cross-entropy loss also may be more 
       appropriate than margin loss in this scenario. -- note i dont think we can do this here - CP

    Parameters
    ----------
    args : (namespace output of argparse.parse_args() - see below for details)
    """
    G = dgl.load_graphs(args.graph_file)[0][0] # load in our graph 

    k = 5
    
    node_features, node_sizes, edge_input_sizes = preprocess_edges(G)

    ep_model = LinkPredictor(edge_input_sizes, 20, 5, G.etypes) ## may need some tweaking
    opt = torch.optim.Adam(ep_model.parameters())

    for epoch in range(100):
        neg_G = construct_negative_graph(G, k, ('chemical', 'chemicalassociateswithdisease', 'disease'))
        pos_score, neg_score = ep_model(G.to(DEVICE), neg_G.to(DEVICE), node_features, ('chemical', 'chemicalassociateswithdisease', 'disease')) # *****

        # margin loss
        loss = compute_lp_loss(pos_score, neg_score)
        opt.zero_grad()
        loss.backward()
        opt.step()
        
        print("epoch: %3d; margin loss: %.5f" % (epoch, loss.item()))

        # Now, we need to figure out something to do with the trained model!

    #ipdb.set_trace()
    
def main(args):
    link_prediction(args)

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Train a heterogeneous RGCN on a link prediction task.")
    parser.add_argument("--graph-file", type=str, default="/Users/cfparis/Desktop/Romano_Rotation/GNN/data/graph.bin",
                        help="File location where the DGL heterograph is stored.")
    parser.add_argument("--lr", type=float, default=0.01,
                        help="Learning rate for the NN optimizer.")
    parser.set_defaults(validation=True)

    args = parser.parse_args()
    main(args)
