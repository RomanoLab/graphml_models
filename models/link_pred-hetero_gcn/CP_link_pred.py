# Link prediction adapted from various sources online and joe's code

import argparse

import dgl
import dgl.function as fn
import dgl.nn as dglnn
import pickle
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
from dgl.dataloading import (
    as_edge_prediction_sampler,
    DataLoader,
    MultiLayerFullNeighborSampler,
    negative_sampler,
    NeighborSampler,
)

DEVICE = 'cpu'

######################################################################################
# PROCESS GRAPH 
######################################################################################
def preprocess_edges(graph):
    '''
    Preprocess graph for link prediction

    Parameters
    ----------
    graph : dgl graph you want to preprocess

    Outputs
    ----------
    node_features : dictionary with each node type and a tensor of ones of length # of number of nodes within that node type 
    node_sizes : dictionary of each node and its tensor length (shape 1) 
    edge_input_sizes : dictionary of each edges with tensor length (shape 1) of the source node corresponding to the edge
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
# GET EDGE SCORES USING RCGN LAYER
######################################################################################
class HeteroRGCNLayer(nn.Module):
    '''
    RCGN LAYER TO CALCULATE NODE REPRESENTATIONS 
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
        Calculate node representation and add as feature to nodes

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
            positive_graph.nodes[srctype].data['Wh_%s' % etype] = Wh # set a feature ‘????’ for all nodes of the source node type
            funcs[etype] = (fn.copy_u('Wh_%s' % etype, 'm'), fn.mean('m', 'h')) # still a bit confused ??????

        positive_graph.multi_update_all(funcs, 'sum') # send messages along all the edges, update the node features of all the nodes
        h = { ntype : positive_graph.nodes[ntype].data['h'] for ntype in positive_graph.ntypes }

        return h

######################################################################################
# SCORE PREDICTOR 
######################################################################################
class HeteroDotProductPredictor(nn.Module):
    '''
    MAKE SCORE PREDICTION USING DOT PRODUCT
    '''
    def forward(self, edge_subgraph, h, etype):
        '''
        Predict edge score and add as feature to edges

        Parameters
        ----------
        edge_subgraph : graph you want to calculate edge scores for
        h : node representation from RGCN
        etype : edge type you want to be making score prediction for

        Outputs
        ----------
        edges_scores : scores for each edge
        '''
        
        # for all edges or just the one we are looking at?
        with edge_subgraph.local_scope():
            edge_subgraph.ndata['h'] = h # get the node representation
            edge_subgraph.apply_edges(fn.u_dot_v('h', 'h', 'score'), etype = etype) # update the features of the specified edge with the provide function (in our case the dot product)
            edges_scores = edge_subgraph.edges[etype].data['score']

            return edges_scores

######################################################################################
# LINK PREDICTION
######################################################################################
class Model(nn.Module):
    '''
    THE LINK PREDICTION MODEL
    '''
    def __init__(self, in_size_dict, hidden_size, out_size, rel_names):
        super().__init__()

        self.sage = HeteroRGCNLayer(in_size_dict, out_size, rel_names)
        self.pred = HeteroDotProductPredictor()

    def forward(self, positive_graph, negative_graph, x, etype):
        '''
        Get scores for positive and negative graph

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

######################################################################################
# COMPUTE LOSS
######################################################################################
def compute_loss(pos_score, neg_score):
    '''
     Get scores for positive and negative graph

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
def training(device, G, model, train_eid_dict):
    '''
    Train link prediction model

    Parameters
    ----------
    device : device we want to use for training
    G : graph we want to train on
    model : our model we will use for training
    train_eid_dict : dictionary of edge types and edge IDs

    Outputs
    ----------
    all_loss : our model loss
    '''

    ################# Sampler and dataloader #########################################
    ## for each source node of each edge in our graph we pick k (in our case 1) destination nodes uniformly 
    # number of layers and how many messages you are taking from each layer could also use: sampler = NeighborSampler([15, 10, 5], prefetch_node_feats=["feat"]) ?????
    sampler = dgl.dataloading.MultiLayerFullNeighborSampler(2) 
    
    # we want to turn our sampler into one that can be used for edge prediction instead of node classification
    sampler = dgl.dataloading.as_edge_prediction_sampler(
        sampler, 
        #exclude = "reverse_id", # exclude edges in the current minibatch but also their reverse edges according to the ID mapping in the argument reverse_eids.
        #reverse_eids = reverse_eids, # reverse edges
        negative_sampler = dgl.dataloading.negative_sampler.Uniform(1) # our negative sampler ***a parameter we can probably tweak here***
        ) 

    ## with a negative sampler, dataloader will generate 3 items per minibatch:
    # a positive graph containing all the edges sampled in the minibatch.
    # a negative graph containing all the non-existent edges generated by the negative sampler.
    # a list of message flow graphs (MFGs) generated by the neighborhood sampler.

    dataloader = dgl.dataloading.DataLoader(
        G, # graph
        train_eid_dict, # dict of edge type and edge ID tensors
        sampler, # sampler
        device = device, # device we want to use 
        batch_size = 1024, # number of indices in each batch ***a parameter we can probably tweak here***
        shuffle = True, # randomly shuffle the indices at each epoch
        drop_last = False, # don't drop the last incomplete batch
        num_workers = 0 # not completely sure what this does i think it is a computer thing
        )
    
    ################# Train model #########################################
    opt = torch.optim.Adam(model.parameters())  # optimization algorithm we want to use

    print('Running epochs...')

    # where we will be storing our loss
    all_loss = []

    # iterates over data loader and feeds in the small graphs + input features to the model defined above
    for epoch in range(10):
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
            pos_score, neg_score = model(positive_graph, negative_graph, sm_node_features, ('chemical', 'chemicalassociateswithdisease', 'disease')) # input features 

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
    parser = argparse.ArgumentParser(description = "Train a heterogeneous RGCN on a link prediction task.")

    # give location of graph file
    parser.add_argument("--graph-file", type=str, default = "/Users/cfparis/Desktop/Romano_Rotation/making_dgl_graph/data/graph.bin",
                        help = "File location where the DGL heterograph is stored.")

    # give location of edge list
    parser.add_argument("--edge-file", type = str, default = "/Users/cfparis/Desktop/Romano_Rotation/making_dgl_graph/data/edge_dict.pkl",
                        help = "File location where the edge list is stored.")

    # not sure what this does
    parser.set_defaults(validation = True) # not sure what this does

    args = parser.parse_args()

    ######################################################################################
    # INITIALIZATION OF GRAPH AND MODEL
    ######################################################################################
    # choose mode to train in
    print("Training in cpu mode.")
    device = torch.device('cpu')
    
    print('Loading in graph...')
    G = dgl.load_graphs(args.graph_file)[0][0]
    G = G.to("cpu")

    print('Getting reverse IDs...')
    with open(args.edge_file, 'rb') as f:
        reverse_eids = pickle.load(f)
    
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
    loss_array = training(device, G, model, train_eid_dict)

    print("Making loss graph...")
    loss_plt = sns.lineplot(data = loss_array)
    loss_fig = loss_plt.figure
    loss_fig.savefig('loss_fig.png')
