## Code to make subgraph in dgl from neo4j csv data
## file outputs dgl graph, networkx graph, edge dictionary as pkl file

##################################################################################
# PACKAGES
##################################################################################
from numpy.core.fromnumeric import size
import torch
import dgl
from dgl import save_graphs, load_graphs, heterograph, edge_type_subgraph
from dgl.data import DGLDataset
import pandas as pd
import numpy as np
import scipy.sparse
import ipdb
from os import path
import networkx as nx

import pickle as pkl

from torch.nn.functional import adaptive_avg_pool2d
##################################################################################

##################################################################################
# FUNCTIONS TO MAKE DGL GRAPH
##################################################################################
## functions
class DGL_GRAPH(DGLDataset):
  def __init__(self, name, file_path, rebuild = False):
    self.rebuild = rebuild
    self.filepath = file_path
    super(DGL_GRAPH, self).__init__(name)

  def process(self):
    print("Processing data...")

    self.read_source_files()
    self.parse_node_features()
    self.process_node_labels()
    self.build_adjacency_matrices()
    self.make_heterograph()
    self.make_edge_dict()
    self.make_networkx_graph()

  def read_source_files(self):
    '''
    Function to read in the source file from neo4j.
    '''
    print("  ...reading source files.")
    # Load node source files
    self.chemicals = pd.read_csv(str(self.filepath) + "chemicals.csv")
    self.genes = pd.read_csv(str(self.filepath) + "genes.csv")
    self.assays = pd.read_csv(str(self.filepath) +  "assays.csv")
    self.diseases = pd.read_csv(str(self.filepath) + "diseases.csv")

    # Load edge source files
    self.chemical_assay = pd.read_csv(str(self.filepath) +  "chemical-assay.csv")
    self.chemical_gene = pd.read_csv(str(self.filepath) + "chemical-gene.csv")
    self.gene_gene = pd.read_csv(str(self.filepath) + "gene-gene.csv")
    self.chemical_disease = pd.read_csv(str(self.filepath) +  "chemical-disease.csv")

  def parse_node_features(self):
    '''
    Function to get node features (only MACCS for chemicals in this case).
    '''
    print("  ...parsing node features.")
    maccs_ndarray = np.empty(shape=(len(self.chemicals), len(self.chemicals.maccs[0])))
    for i, m in enumerate(self.chemicals.maccs):
      maccs_ndarray[i,:] = [int(mm) for mm in m]
    self.maccs_tensor = torch.tensor(maccs_ndarray, dtype=torch.bool)

  def make_rel_t(self, s_node, rel, o_node):
    '''
    Make a tensor representing links between a subject and an object node
    type for a specified edge type.

    Parameters
    ----------
    s_node : subject node (str)
    rel : relation (str)
    o_node : object node (str)
    '''
    
    # Dynamically retrieve dataframes based on subject and object names
    s_df = eval("self."+s_node+"s")
    rel_df = eval("self."+s_node+"_"+o_node)
    o_df = eval("self."+o_node+"s")

    filtered_rels = rel_df.loc[rel_df.edge == rel,:]

    s_idx_map = dict(zip(s_df.node, s_df.index.tolist()))
    o_idx_map = dict(zip(o_df.node, o_df.index.tolist()))

    s_conv = [s_idx_map[x] for x in filtered_rels.node1.values]  # 'node1' is subject
    o_conv = [o_idx_map[x] for x in filtered_rels.node2.values]  # 'node2' is object

    s = torch.tensor(s_conv)
    o = torch.tensor(o_conv)

    return (s, o)

  def process_node_labels(self):
    '''
    Function to label nodes (used for node classification) and dump pkl file.
    '''
    print("  ...processing node labels.")
    inactive_assays = self.make_rel_t('chemical', 'CHEMICALHASINACTIVEASSAY', 'assay') 
    active_assays = self.make_rel_t('chemical', 'CHEMICALHASACTIVEASSAY', 'assay')

    active_assay_mask = active_assays[1] == 47
    inactive_assay_mask = inactive_assays[1] == 47
    
    # drops labels into pkl files
    active_assay_nodes = active_assays[0][active_assay_mask]
    inactive_assay_nodes = inactive_assays[0][inactive_assay_mask]
    pkl.dump(active_assay_nodes, open(str(self.filepath) + "active_assays.pkl", 'wb'))
    pkl.dump(inactive_assay_nodes, open(str(self.filepath) + "inactive_assays.pkl", 'wb')) 
  
  def make_adjacency(self, s_node, rel, o_node):
    '''
    Function to make an adjacency list for a specific metaedge.

    Parameters
    ----------
    s_node : str
    rel : str
    o_node : str

    Outputs
    ----------
    adjacency matrix as scipy sparse matrix

    '''
    s_df = eval("self."+s_node+"s")
    rel_df = eval("self."+s_node+"_"+o_node)
    o_df = eval("self."+o_node+"s")
    
    filtered_rels = rel_df.loc[rel_df.edge == rel,:]

    s_idx_map = dict(zip(s_df.node, s_df.index.tolist()))
    o_idx_map = dict(zip(o_df.node, o_df.index.tolist()))

    s_conv = [s_idx_map[x] for x in filtered_rels.node1.values]  # 'node1' is subject
    o_conv = [o_idx_map[x] for x in filtered_rels.node2.values]  # 'node2' is object

    adj = scipy.sparse.lil_matrix( (max(s_conv)+1, max(o_conv)+1) )  # Add 1 for zero-based indexing

    for i in range(len(s_conv)):
      adj[s_conv[i],o_conv[i]] = 1

    return scipy.sparse.csr_matrix(adj)

  def build_adjacency_matrices(self):
    '''
    Function to make adjacency matrices.
    '''
    print("  ...constructing adjacency matrices.")
    metaedges = {
      'chemicalhasactiveassay': ('chemical', 'CHEMICALHASACTIVEASSAY', 'assay'),
      'chemicalhasinactiveassay': ('chemical', 'CHEMICALHASINACTIVEASSAY', 'assay'),
      'chemicalbindsgene': ('chemical', 'CHEMICALBINDSGENE', 'gene'),
      'chemicaldecreasesexpression': ('chemical', 'CHEMICALDECREASESEXPRESSION', 'gene'),
      'chemicalincreasesexpression': ('chemical', 'CHEMICALINCREASESEXPRESSION', 'gene'),
      'geneinteractswithgene': ('gene', 'GENEINTERACTSWITHGENE', 'gene'), 
      'chemicalassociateswithdisease': ('chemical', 'CHEMICALASSOCIATESWITHDISEASE', 'disease')
    }

    self.adjacency = dict()

    for k, (s,r,o) in metaedges.items():
      self.adjacency[(s,r,o)] = self.make_adjacency(s, r, o)

      # if you want to print each individual matrix
      #scipy.sparse.save_npz("adj_matrices" + str(k) + ".npz", self.make_adjacency(s, r, o))

  def make_heterograph(self):
    '''
    Function to make dgl heterograph.
    '''
    print("  ...finalizing heterogeneous graph.")
    graph_data = {
      ('chemical', 'chemicalhasinactiveassay', 'assay'): self.adjacency[('chemical', 'CHEMICALHASINACTIVEASSAY', 'assay')].nonzero(),
      ('assay', 'assayinactiveforchemical', 'chemical'): self.adjacency[('chemical', 'CHEMICALHASINACTIVEASSAY', 'assay')].transpose().nonzero(),

      ('chemical', 'chemicalhasactiveassay', 'assay'): self.adjacency[('chemical', 'CHEMICALHASACTIVEASSAY', 'assay')].nonzero(),
      ('assay', 'assayactiveforchemical', 'chemical'): self.adjacency[('chemical', 'CHEMICALHASACTIVEASSAY', 'assay')].transpose().nonzero(),

      ('chemical', 'chemicalbindsgene', 'gene'): self.adjacency[('chemical', 'CHEMICALBINDSGENE', 'gene')].nonzero(),
      ('gene', 'genebindedbychemical', 'chemical'): self.adjacency[('chemical', 'CHEMICALBINDSGENE', 'gene')].transpose().nonzero(),

      ('chemical', 'chemicaldecreasesexpression', 'gene'): self.adjacency[('chemical', 'CHEMICALDECREASESEXPRESSION', 'gene')].nonzero(),
      ('gene', 'expressiondecreasedbychemical', 'chemical'): self.adjacency[('chemical', 'CHEMICALDECREASESEXPRESSION', 'gene')].transpose().nonzero(),

      ('chemical', 'chemicalincreasesexpression', 'gene'): self.adjacency[('chemical', 'CHEMICALINCREASESEXPRESSION', 'gene')].nonzero(),
      ('gene', 'expressionincreasedbychemical', 'chemical'): self.adjacency[('chemical', 'CHEMICALINCREASESEXPRESSION', 'gene')].transpose().nonzero(),

      ('gene', 'geneinteractswithgene', 'gene'): self.adjacency[('gene', 'GENEINTERACTSWITHGENE', 'gene')].nonzero(),
      ('gene', 'geneinverseinteractswithgene', 'gene'): self.adjacency[('gene', 'GENEINTERACTSWITHGENE', 'gene')].transpose().nonzero(),

      ('chemical', 'chemicalassociateswithdisease', 'disease'): self.adjacency[('chemical', 'CHEMICALASSOCIATESWITHDISEASE', 'disease')].nonzero(),
      ('disease', 'diseaseassociateswithchemical', 'chemical'): self.adjacency[('chemical', 'CHEMICALASSOCIATESWITHDISEASE', 'disease')].transpose().nonzero()
    }

    self.G = heterograph(graph_data)

  def make_edge_dict(self):
    '''
    Function to make a dictionary of the edges and reverse edges in graph (helpful for link prediction).
    We want the dictionary to be in the format edge:reverse edge for each edge. 
    Dumps to pkl file.
    '''
    print('  ...making edge dictionary')

    edge_dict = {'chemicalhasinactiveassay': 'assayinactiveforchemical', 'assayinactiveforchemical': 'chemicalhasinactiveassay', 
                 'chemicalhasactiveassay': 'assayactiveforchemical', 'assayactiveforchemical': 'chemicalhasactiveassay', 
                 'chemicalbindsgene': 'genebindedbychemical', 'genebindedbychemical': 'chemicalbindsgene', 
                 'chemicaldecreasesexpression': 'expressiondecreasedbychemical', 'expressiondecreasedbychemical': 'chemicaldecreasesexpression', 
                 'chemicalincreasesexpression': 'expressionincreasedbychemical', 'expressionincreasedbychemical': 'chemicalincreasesexpression', 
                 'geneinteractswithgene': 'geneinverseinteractswithgene', 'geneinverseinteractswithgene': 'geneinteractswithgene', 
                 'chemicalassociateswithdisease': 'diseaseassociateswithchemical', 'diseaseassociateswithchemical': 'chemicalassociateswithdisease'
                 }

    # dumps to pkl file
    pkl.dump(edge_dict, open(str(self.filepath) + "edge_dict.pkl", 'wb'))

  def make_networkx_graph(self):
    '''
    Function to make networkx graph.
    '''
    print('  ...making networkx graph')
    hg = dgl.to_homogeneous(self.G) # to homogenous graph 
    self.nx_g = dgl.to_networkx(hg, node_attrs = ['_TYPE'], edge_attrs = ['_TYPE']) # to networkx graph

  def save(self):
    print("Saving data...")
    output_filename = path.join("data", "graph.bin")

    # save dgl graph
    save_graphs(output_filename, self.G)

    # save networkx graph
    nx.write_edgelist(self.nx_g, "networkx.edgelist") # write as edgelist

if __name__=="__main__":
  dset = DGL_GRAPH(name = "GNN", file_path = '/Users/cfparis/Desktop/Romano_Rotation/making_dgl_graph/data/')