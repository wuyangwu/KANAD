import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import Sequential as Seq
from models.utils import Decoder
from models.gcn_lib.torch_edge import DenseDilatedKnnGraph
from models.gcn_lib.torch_vertex import GraphConv2d
from models.gcn_lib.torch_nn import get_act_layer
from timm.models.layers import DropPath
class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size
    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()

class ConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_sizes, dilation=2, device="cpu", dropout=0, pooling=True, **kwargs):
        super(ConvNet, self).__init__()
        layers = []
        for i in range(len(kernel_sizes)):
            dilation_size = dilation ** i
            kernel_size = kernel_sizes[i]
            padding = (kernel_size-1) * dilation_size
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [nn.Conv1d(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size, padding=padding), nn.ReLU(), Chomp1d(padding), nn.Dropout(dropout)]
            
        self.network = nn.Sequential(*layers)
        
        self.pooling = pooling
        if self.pooling:
            self.maxpool = nn.MaxPool1d(num_channels[-1]).to(device)
        self.network.to(device)
        
    
    def forward(self, x): #[batch_size, T, 1]
        x = x.permute(0, 2, 1) #[batch_size, 1, T]
        out = self.network(x) #[batch_size, out_dim, T]
        out = out.permute(0, 2, 1) #[batch_size, T, out_dim]
        if self.pooling:
            return self.maxpool(out)
        else:
            return out

class FFN(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act='relu', drop_path=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Sequential(
            nn.Conv2d(in_features, hidden_features, 1, stride=1, padding=0),
            nn.BatchNorm2d(hidden_features),
        )
        self.act = get_act_layer(act)
        self.fc2 = nn.Sequential(
            nn.Conv2d(hidden_features, out_features, 1, stride=1, padding=0),
            nn.BatchNorm2d(out_features),
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        # shortcut = x
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.drop_path(x)
        # x = self.drop_path(x) + shortcut
        return x
        
class InnerEncoder(nn.Module): 
    def __init__(self, input_size, device, **kwargs):
        super(InnerEncoder, self).__init__()
        # temoral = [256,256,128]
        # kernel_sizes = [3,3,3]
        temporal_dims = kwargs["inner_hidden_sizes"]
        kernel_sizes = kwargs["inner_kernel_sizes"]
        dropout = kwargs["inner_dropout"]
        
        assert len(temporal_dims) == len(kernel_sizes)
        temporal_dims[-1] = kwargs["hidden_size"]
        # input_size = 4, temperal_dims
        self.net = ConvNet(input_size, temporal_dims, kernel_sizes, device=device, dropout=dropout, pooling=False)

    def forward(self, x: torch.Tensor): #[batch_size, T, var_num / metric_num]
        return self.net(x)

class SignalGraphConv2d(GraphConv2d):
    """
    Dynamic graph convolution layer
    """

    def __init__(self, in_channels, out_channels, kernel_size=9, dilation=1, conv='edge', act='relu',
                 norm=None, bias=True, stochastic=False, epsilon=0.0):
        super(SignalGraphConv2d, self).__init__(in_channels, out_channels, conv, act, norm, bias)
        self.dilated_knn_graph = DenseDilatedKnnGraph(kernel_size, dilation, stochastic, epsilon)

    def forward(self, x, relative_pos=None, **kwargs):
        tmp = x
        y = None
        # x = [128,128,10,1]
        # edge_index = [2,128,128,4]
        edge_index = self.dilated_knn_graph(x, y, relative_pos)
        x = super(SignalGraphConv2d, self).forward(x, edge_index, y)
        return tmp + x

class KpiEncoder(nn.Module):
    def __init__(self, var_nums, device, kpi_architect="by_aspect", **kwargs):
        super(KpiEncoder, self).__init__()
        self.kpi_type = kpi_architect
        self.var_nums = var_nums
        self.metric_num = sum(var_nums)
        self.group_num = len(var_nums)

        self.window_size = kwargs["window_size"]

        if self.kpi_type == "by_knn":
            k = kwargs['top_k']  # 4
            self.token_embed = nn.Conv1d(in_channels=self.metric_num, out_channels=128, kernel_size=3,
                                         padding_mode="circular", padding=1)
            self.i_encoder = InnerEncoder(self.metric_num, device, **kwargs)
            self.graph_backbone = nn.ModuleList([])
            blocks = [1,1] # [1, 1]
            self.n_blocks = sum(blocks)
            conv = kwargs['Graph_conv']
            act = 'gelu'# 'gelu'
            norm = 'batch'  # 'batch'
            bias = 'True'  # True
            epsilon = '0.2'  # 0.p2   stochastic epsilon for gcn
            stochastic = False # False
            num_knn = [int(x.item()) for x in torch.linspace(k, k, self.n_blocks)]  #
            max_dilation = max(1, 25 // max(num_knn)) #
            idx = 0
            dpr = [x.item() for x in torch.linspace(0, 0, self.n_blocks)]
            # dpr = [0.1,0.1]
            for i in range(len(blocks)):
                for j in range(blocks[i]):
                    self.graph_backbone += [
                        Seq(SignalGraphConv2d(128, 128, num_knn[idx],
                                              min(idx // 4 + 1, max_dilation),
                                              conv, act, norm, bias, stochastic, epsilon),
                            FFN(128, 128, act=act, drop_path=dpr[idx]))]
                    idx += 1
            self.graph_backbone = Seq(*self.graph_backbone)
        else: raise ValueError("Unrecognized Kpi Architect Type {}!".format(self.kpi_type))

    def forward(self, ts): #ts group list
        batch_size = ts[0].size(0)
        d = ts[0].device
        if self.kpi_type == "by_knn":
            data = [torch.tensor(t).float() for t in ts]
            data = torch.cat(data, dim=1)
            data = data.to(d)
            inner_input = data.permute(0, 2, 1).float()
            encoderdata = self.i_encoder(inner_input)
            encoderdata = encoderdata.permute(0,2,1)
            encoderdata = encoderdata.unsqueeze(3)
            for i in range(len(self.graph_backbone)):
                encoderdata = self.graph_backbone[i](encoderdata)
            encoderdata = encoderdata.squeeze(-1)
            return encoderdata.permute(0, 2, 1)
from torch.nn.functional import softmax as sf
class KpiModel(nn.Module):
    def __init__(self, var_nums, device, **kwargs):
        super(KpiModel, self).__init__() #init BaseModel
        self.var_nums = var_nums
        self.encoder = KpiEncoder(var_nums, device, **kwargs)
        self.decoder = Decoder(kwargs["hidden_size"], kwargs["window_size"], **kwargs)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, input_dict, flag=False):
        y = input_dict["label"].long().view(-1) #[batch_size, ]    
        
        kpi_re = self.encoder(input_dict["kpi_features"])
        logits = self.decoder(kpi_re) #[batch_size, 2]
        if y.size(0) == 1: logits = logits.unsqueeze(0)
        if flag:
            y_pred = logits.detach().cpu().numpy().argmax(axis=1)
            conf = sf(logits.detach().cpu(), dim=1).numpy().max(axis=1) #[bz, 1]
            return {"y_pred": y_pred, "conf": conf}

        loss = self.criterion(logits, y)
        return {"loss": loss}
            
