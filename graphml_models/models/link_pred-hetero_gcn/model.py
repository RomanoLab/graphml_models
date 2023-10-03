from graphml_models.data import loader

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.loader import DataLoader

from ogb.linkproppred import Evaluator

import ipdb


class KGEModel(nn.Module):
    def __init__(self, nentity, nrelation, evaluator, gamma=12.0, hidden_dim=500):
        super(KGEModel, self).__init__()
        self.model_name = 'TransE'
        self.nentity = nentity
        self.nrelation = nrelation
        self.hidden_dim = hidden_dim
        self.epsilon = 2.0

        self.gamma = nn.Parameter(
            torch.Tensor([gamma]),
            requires_grad = False
        )

        self.embedding_range = nn.Parameter(
            torch.Tensor([(self.gamma.item() + self.epsilon) / hidden_dim]),
            requires_grad = False
        )

        self.entity_dim = hidden_dim
        self.relation_dim = hidden_dim

        self.entity_embedding = nn.Parameter(torch.zeros(nentity, self.entity_dim))
        nn.init.uniform_(
            tensor=self.entity_embedding,
            a=-self.embedding_range.item(),  # Notice negative value
            b=self.embedding_range.item()
        )

        self.relation_embedding = nn.Parameter(torch.zeros(nrelation, self.relation_dim))
        nn.init.uniform_(
            tensor=self.relation_embedding,
            a=-self.embedding_range.item(),  # Notice negative value
            b=self.embedding_range.item()
        )

    def forward(self):
        pass

    def TransE(self, head, relation, tail, mode):
        if mode == 'head-batch':
            score = head * (relation * tail)
        else:
            score = (head + relation) - tail
        
        score = self.gamma.item() - torch.norm(score, p=1, dim=2)
        return score
    
    @staticmethod
    def train_step(model, optimizer, train_iterator, args):
        model.train()
        optimizer.zero_grad()
        positive_sample, negative_sample, subsampling_weight, mode = next(train_iterator)

        negative_score = model((positive_sample, negative_sample), mode=mode)
        negative_score = F.logsigmoid(-negative_score).mean(dim = 1)

        positive_score = model(positive_sample)
        positive_score = F.logsigmoid(positive_score).squeeze(dim = 1)



class HeteroGCN():
    def __init__(self, dataset):
        split_edge = dataset.get_edge_split()
        self.train_triples = split_edge["train"]
        self.valid_triples = split_edge["valid"]
        self.test_triples = split_edge["test"]

        self.train_loader = DataLoader(self.train_triples, batch_size=32, shuffle=True)
        self.valid_loader = DataLoader(self.valid_triples, batch_size=32, shuffle=False)
        self.test_loader = DataLoader(self.test_triples, batch_size=32, shuffle=False)
        
        self.nrelation = int(max(self.train_triples['relation']))+1

        self.entity_dict = dict()
        cur_idx = 0
        for key in dataset[0]['num_nodes_dict']:
            self.entity_dict['key'] = (cur_idx, cur_idx+dataset[0]['num_nodes_dict'][key])
            cur_idx += dataset[0]['num_nodes_dict'][key]
        self.nentity = sum(dataset[0]['num_nodes_dict'].values())

        self.evaluator = Evaluator(name = "ogbl-biokg")

        self.model = KGEModel(
            self.nentity,
            self.nrelation,
            self.evaluator
        )

        for name, param in self.model.named_parameters():
            print(f"Parameter {name}: {str(param.size())}, require_grad = {str(param.requires_grad)}")

    def fit(self):
        print("Training network on biokg...")

        #train_dataloader_head = DataLoader()

    def predict(self):
        pass

def main():
    dataset = loader.load_graph(graph_name='ogbl-biokg', format='pyg')
    
    network = HeteroGCN(dataset)
    network.fit()

    ipdb.set_trace()

if __name__ == "__main__":
    main()