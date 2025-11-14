import dgl
import torch
from torch import nn
from torch.autograd import Variable
from models.utils import SelfAttention, Embedder, Decoder, Trans
from dgl.nn.pytorch.conv import SAGEConv


class GCNLog(nn.Module):
    def __init__(self, embedding_size, h_feats, dropout):
        super(GCNLog, self).__init__()
        self.gcn_out_dim = 4 * h_feats
        # self.embedding = nn.Embedding(256 + 1, embedding_size)
        self.gcn1 = SAGEConv(embedding_size, h_feats, 'mean', feat_drop=dropout, activation=nn.PReLU(h_feats), norm=nn.BatchNorm1d(h_feats))
        self.gcn2 = SAGEConv(h_feats, h_feats, 'mean', feat_drop=dropout, activation=nn.PReLU(h_feats), norm=nn.BatchNorm1d(h_feats))
        self.gcn3 = SAGEConv(h_feats, h_feats, 'mean', feat_drop=dropout, activation=nn.PReLU(h_feats), norm=nn.BatchNorm1d(h_feats))
        self.gcn4 = SAGEConv(h_feats, h_feats, 'mean', activation=nn.PReLU(h_feats), norm=nn.BatchNorm1d(h_feats))

    def forward(self, g, h):
        # in_feat = in_feat.long()
        # h = self.embedding(in_feat.view(-1))
        h1 = self.gcn1(g, h)
        h2 = self.gcn2(g, h1)
        h3 = self.gcn3(g, h2)
        h4 = self.gcn4(g, h3)
        g.ndata['h'] = torch.cat((h1, h2, h3, h4), dim=1)
        g_vec = dgl.mean_nodes(g, 'h')
        return g_vec
class LogGraphEncoder(nn.Module):
    def __init__(self,device,log_dropout=0.2,**kwargs):
        super(LogGraphEncoder, self).__init__()
        self.hidden = kwargs["hidden_size"]  # 128
        self.window = kwargs["window_size"]  # 10
        embedding_dim = kwargs["word_embedding_dim"]  # 32
        self.log_header_graphConv = GCNLog(embedding_size=embedding_dim, h_feats=self.hidden, dropout=log_dropout)
        self.linear = nn.Linear(512, 1280)
    def forward(self, logGraph_data):
        batch_size = logGraph_data.batch_size
        header_gcn_out = self.log_header_graphConv(logGraph_data, logGraph_data.ndata['feat'])
        log_re = self.linear(header_gcn_out).reshape(batch_size,self.window,-1)
        return log_re #[batch_size, window_size, hidden_size]

from torch.nn.functional import softmax as sf
class LogModel(nn.Module):
    def __init__(self, device, vocab_size=300, **kwargs):
        super(LogModel, self).__init__()
        self.feature_type = kwargs["feature_type"]
        if "word2vec" not in self.feature_type:
            self.embedder = Embedder(vocab_size, **kwargs)

        self.encoder = LogGraphEncoder(device, **kwargs)
        self.decoder = Decoder(kwargs["hidden_size"], kwargs["window_size"], **kwargs)
        
        self.criterion = nn.CrossEntropyLoss()
    
    def forward(self, input_dict, flag=False):
        y = input_dict["label"].long().view(-1)
        log_x = input_dict["log_features"]
        if "word2vec" not in self.feature_type: 
            log_x = self.embedder(log_x)           
        
        log_re = self.encoder(log_x) #[batch_size, W, hidden_size*dir_num]
        logits = self.decoder(log_re)
        if y.size(0) == 1: logits = logits.unsqueeze(0)

        if flag:
            y_pred = logits.detach().cpu().numpy().argmax(axis=1)
            conf = sf(logits.detach().cpu(), dim=1).numpy().max(axis=1) #[bz, 1]
            return {"y_pred": y_pred, "conf": conf}

        loss = self.criterion(logits, y)
        return {"loss": loss}
