import torch
from torch import nn
from torch.autograd import Variable
import numpy as np
import math
# from gcn_lib.torch_nn import get_act_layer
import torch.nn.functional as F

class Decoder(nn.Module):
    def __init__(self, encoded_dim, T, **kwargs):
        super(Decoder, self).__init__()
        linear_size = kwargs["linear_size"]

        layers = []
        for i in range(kwargs["decoder_layer_num"]-1): #decoder_layer_num =2
            input_size = encoded_dim if i == 0 else linear_size
            layers += [nn.Linear(input_size, linear_size),nn.ReLU()]
        layers += [KANLinear(linear_size, 2)]
        self.net = nn.Sequential(*layers)
        self.self_attention = kwargs["self_attention"]
        if self.self_attention:
            self.attn = SelfAttention(encoded_dim, T)

    def forward(self, x: torch.Tensor): #[batch_size, T, hidden_size*dir_num]
        if self.self_attention: ret = self.attn(x)
        else: ret = x[:, -1, :]
        return self.net(ret)

class Decoder_mult(nn.Module):
    def __init__(self):
        super(Decoder_mult, self).__init__()
        layers = []
        for i in range(2):
            input_size = 256*10 if i ==0 else 512
            layers += [nn.Linear(input_size,512), nn.ReLU()]
        # layers += [nn.Linear(512, 2)]
        layers += [KANLinear(512, 2)]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor):
        B, _, _ = x.size()
        ret = x.contiguous().view(B, -1)

        return self.net(ret)

class Embedder(nn.Module):
    def __init__(self, vocab_size=300, **kwargs):
        super(Embedder, self).__init__()
        self.embedding_dim = kwargs["word_embedding_dim"]
        self.embedder = nn.Embedding(vocab_size, self.embedding_dim, padding_idx=0)
    
    def forward(self, x): #[batch_size, T, word_lst_num]
        return self.embedder(x.long())

class SelfAttention(nn.Module):
    def __init__(self, input_size, seq_len):
        """
        Args:
            input_size: int, hidden_size * num_directions
            seq_len: window_size
        """
        super(SelfAttention, self).__init__()
        self.atten_w = nn.Parameter(torch.randn(seq_len, input_size, 1))
        self.atten_bias = nn.Parameter(torch.randn(seq_len, 1, 1))
        self.glorot(self.atten_w)
        self.atten_bias.data.fill_(0)

    def forward(self, x):
        # x: [batch_size, window_size, 2*hidden_size]
        input_tensor = x.transpose(1, 0)  # w x b x h
        input_tensor = (torch.bmm(input_tensor, self.atten_w) + self.atten_bias)  # w x b x out
        input_tensor = input_tensor.transpose(1, 0)
        atten_weight = input_tensor.tanh()
        weighted_sum = torch.bmm(atten_weight.transpose(1, 2), x).squeeze()
        return weighted_sum

    def glorot(self, tensor):
        if tensor is not None:
            stdv = math.sqrt(6.0 / (tensor.size(-2) + tensor.size(-1)))
            tensor.data.uniform_(-stdv, stdv)

class Trans(nn.Module):
    def __init__(self, input_size, layer_num, out_dim, dim_feedforward=512, dropout=0, device="cpu", norm=None, nhead=8):
        super(Trans, self).__init__()
        encoder_layer = nn.TransformerEncoderLayer(
                            d_model=input_size, 
                            dim_feedforward=dim_feedforward, #default: 2048
                            nhead=nhead, 
                            dropout=dropout, 
                            batch_first=True)
        self.net = nn.TransformerEncoder(encoder_layer, num_layers=layer_num, norm=norm).to(device)
        self.out_layer = nn.Linear(input_size, out_dim)
    def forward(self, x: torch.Tensor): #[batch_size, T, var]
        out = self.net(x)
        return self.out_layer(out)

class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V, attn_mask,d_k=64):                      # Q: [batch_size, n_heads, len_q, d_k]
                                                                       # K: [batch_size, n_heads, len_k, d_k]
                                                                       # V: [batch_size, n_heads, len_v(=len_k), d_v]
                                                                       # attn_mask: [batch_size, n_heads, seq_len, seq_len]
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k)   # scores : [batch_size, n_heads, len_q, len_k]
        scores.masked_fill_(attn_mask, -1e9)                           # 如果时停用词P就等于 0
        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V)                                # [batch_size, n_heads, len_q, d_v]
        return context, attn
        
class Cross_Gated_Info_Filter(nn.Module):
    def __init__(self, in_size,window_size):
        super(Cross_Gated_Info_Filter, self).__init__()
        self.filter1 = nn.Sequential(
            nn.Linear(in_size, in_size),
            nn.PReLU(window_size),
            nn.Linear(in_size, in_size)
        )
        self.filter2 = nn.Sequential(
            nn.Linear(in_size, in_size),
            nn.PReLU(window_size),
            nn.Linear(in_size, in_size)
        )

    def forward(self, x, y):
        ori_x = x
        ori_y = y
        z1 = self.filter1(x).sigmoid() * ori_y
        z2 = self.filter2(y).sigmoid() * ori_x
        return torch.cat([z1, z2], dim=-1)

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model=512, n_heads=8, d_k='none', d_v='none', device="cpu"):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.device = device
        self.d_k = d_k if d_k != 'none' else int(d_model / n_heads)
        self.d_v = d_v if d_v != 'none' else int(d_model / n_heads)
        self.W_Q = nn.Linear(d_model, self.d_k * self.n_heads, bias=False)
        self.W_K = nn.Linear(d_model, self.d_k * self.n_heads, bias=False)
        self.W_V = nn.Linear(d_model, self.d_v * self.n_heads, bias=False)
        # self.fc = nn.Linear(n_heads * self.d_v, self.n_heads, bias=False)
        # self.fc = nn.Linear(d_model, d_model, bias=False)

    def forward(self, query, context, attn_mask='none'):
        input_Q = query
        input_K = context
        input_V = query
        # input_Q: [batch_size, len_q, d_model]
        # input_K: [batch_size, len_k, d_model]
        # input_V: [batch_size, len_v(=len_k), d_model]
        # attn_mask: [batch_size, seq_len, seq_len]
        residual, batch_size = input_Q, input_Q.size(0)
        Q = self.W_Q(input_Q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1,
                                                                                     2)  # Q: [batch_size, n_heads, len_q, d_k]
        K = self.W_K(input_K).view(batch_size, -1, self.n_heads, self.d_k).transpose(1,
                                                                                     2)  # K: [batch_size, n_heads, len_k, d_k]
        V = self.W_V(input_V).view(batch_size, -1, self.n_heads, self.d_v).transpose(1,
                                                                                     2)  # V: [batch_size, n_heads, len_v(=len_k), d_v]
        attn_mask = attn_mask.unsqueeze(1).repeat(1, self.n_heads, 1, 1) if attn_mask != 'none' else torch.zeros(
            batch_size, self.n_heads, Q.size(2), K.size(2)).bool().to(
            self.device)  # attn_mask : [batch_size, n_heads, seq_len, seq_len]
        context, attn = ScaledDotProductAttention()(Q, K, V, attn_mask,
                                                    d_k=self.d_k)  # context: [batch_size, n_heads, len_q, d_v]
        # attn: [batch_size, n_heads, len_q, len_k]
        context = context.transpose(1, 2).reshape(batch_size, -1,
                                                  self.n_heads * self.d_v)  # context: [batch_size, len_v, n_heads * d_v]
        # output = self.fc(context)                                                # [batch_size, len_v, d_model]
        output = context
        return nn.LayerNorm(self.d_model).cuda()(output + residual), attn

class Attention_mask(nn.Module):
    def __init__(self):
        super(Attention_mask, self).__init__()

    def forward(self, x):
        xsum = torch.sum(x, dim=2, keepdim=True)
        # xsum = torch.sum(xsum, dim=3, keepdim=True)
        xshape = tuple(x.size())
        return x / xsum * xshape[1] * xshape[2] * 0.5


class ROI_Selection_Block(nn.Module):

    def __init__(self, in_channel, out_channel, kernel_size=3, stride=1, padding=1, act="relu", device="cpu",dropout_rate=0.2):
        super(ROI_Selection_Block, self).__init__()
        self.data_conv = nn.Sequential(
            nn.Linear(in_channel, out_channel, bias=False, device=device),
            # nn.Conv1d(in_channel, out_channel, kernel_size, stride=1, padding=padding, device=device),
            nn.LayerNorm(256),
            # nn.Linear(in_channel, out_channel, bias=False,device=device),
            # nn.Conv1d(in_channel, out_channel, kernel_size, stride=1, padding=padding, device=device),
            # get_act_layer(act),
        )
        self.attn_conv = nn.Sequential(
            nn.Conv1d(in_channel, out_channel, kernel_size, stride=1, padding=padding,device=device),
            nn.Sigmoid(),
        )
        self.attn_mask = Attention_mask()
        self.drop = nn.Dropout(dropout_rate)
        self.out_conv = nn.Sequential(
            nn.Conv1d(in_channel, out_channel, kernel_size, stride=stride, padding=padding,device=device),
            nn.Tanh(),
        )

    def forward(self, inputs):  # B, T, w
        # conv_out = self.data_conv(inputs)
        conv_out = inputs.permute(0, 2, 1)
        attent = self.attn_conv(conv_out)
        attent = attent.permute(0, 2, 1)

        attent_mask = self.attn_mask(attent)
        conv_out = conv_out.permute(0, 2, 1)
        out = conv_out * attent_mask
        out = out.permute(0, 2, 1)
        out = self.drop(out)

        out = self.out_conv(out)
        out = out.permute(0, 2, 1)
        return out

class KANLinear(torch.nn.Module):
    def __init__(
            self,
            in_features,
            out_features,
            grid_size=5,
            spline_order=3,
            scale_noise=0.1,
            scale_base=1.0,
            scale_spline=1.0,
            enable_standalone_scale_spline=True,
            base_activation=torch.nn.SiLU,
            grid_eps=0.02,
            grid_range=[-1, 1],
    ):
        super(KANLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order

        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = (
            (
                    torch.arange(-spline_order, grid_size + spline_order + 1) * h
                    + grid_range[0]
            )
            .expand(in_features, -1)
            .contiguous()
        )
        self.register_buffer("grid", grid)

        self.base_weight = torch.nn.Parameter(torch.Tensor(out_features, in_features))
        self.spline_weight = torch.nn.Parameter(
            torch.Tensor(out_features, in_features, grid_size + spline_order)
        )
        if enable_standalone_scale_spline:
            self.spline_scaler = torch.nn.Parameter(
                torch.Tensor(out_features, in_features)
            )

        self.scale_noise = scale_noise
        self.scale_base = scale_base
        self.scale_spline = scale_spline
        self.enable_standalone_scale_spline = enable_standalone_scale_spline
        self.base_activation = base_activation()
        self.grid_eps = grid_eps

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5) * self.scale_base)
        with torch.no_grad():
            noise = (
                    (
                            torch.rand(self.grid_size + 1, self.in_features, self.out_features)
                            - 1 / 2
                    )
                    * self.scale_noise
                    / self.grid_size
            )
            self.spline_weight.data.copy_(
                (self.scale_spline if not self.enable_standalone_scale_spline else 1.0)
                * self.curve2coeff(
                    self.grid.T[self.spline_order: -self.spline_order],
                    noise,
                )
            )
            if self.enable_standalone_scale_spline:
                # torch.nn.init.constant_(self.spline_scaler, self.scale_spline)
                torch.nn.init.kaiming_uniform_(self.spline_scaler, a=math.sqrt(5) * self.scale_spline)

    def b_splines(self, x: torch.Tensor):
        """
        Compute the B-spline bases for the given input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).

        Returns:
            torch.Tensor: B-spline bases tensor of shape (batch_size, in_features, grid_size + spline_order).
        """
        assert x.dim() == 2 and x.size(1) == self.in_features

        grid: torch.Tensor = (
            self.grid
        )  # (in_features, grid_size + 2 * spline_order + 1)
        x = x.unsqueeze(-1)
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)
        for k in range(1, self.spline_order + 1):
            bases = (
                            (x - grid[:, : -(k + 1)])
                            / (grid[:, k:-1] - grid[:, : -(k + 1)])
                            * bases[:, :, :-1]
                    ) + (
                            (grid[:, k + 1:] - x)
                            / (grid[:, k + 1:] - grid[:, 1:(-k)])
                            * bases[:, :, 1:]
                    )

        assert bases.size() == (
            x.size(0),
            self.in_features,
            self.grid_size + self.spline_order,
        )
        return bases.contiguous()

    def curve2coeff(self, x: torch.Tensor, y: torch.Tensor):
        """
        Compute the coefficients of the curve that interpolates the given points.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).
            y (torch.Tensor): Output tensor of shape (batch_size, in_features, out_features).

        Returns:
            torch.Tensor: Coefficients tensor of shape (out_features, in_features, grid_size + spline_order).
        """
        assert x.dim() == 2 and x.size(1) == self.in_features
        assert y.size() == (x.size(0), self.in_features, self.out_features)

        A = self.b_splines(x).transpose(
            0, 1
        )  # (in_features, batch_size, grid_size + spline_order)
        B = y.transpose(0, 1)  # (in_features, batch_size, out_features)
        solution = torch.linalg.lstsq(
            A, B
        ).solution  # (in_features, grid_size + spline_order, out_features)
        result = solution.permute(
            2, 0, 1
        )  # (out_features, in_features, grid_size + spline_order)

        assert result.size() == (
            self.out_features,
            self.in_features,
            self.grid_size + self.spline_order,
        )
        return result.contiguous()

    @property
    def scaled_spline_weight(self):
        return self.spline_weight * (
            self.spline_scaler.unsqueeze(-1)
            if self.enable_standalone_scale_spline
            else 1.0
        )

    def forward(self, x: torch.Tensor):
        assert x.size(-1) == self.in_features
        original_shape = x.shape
        x = x.view(-1, self.in_features)

        base_output = F.linear(self.base_activation(x), self.base_weight)
        spline_output = F.linear(
            self.b_splines(x).view(x.size(0), -1),
            self.scaled_spline_weight.view(self.out_features, -1),
        )
        output = base_output + spline_output

        output = output.view(*original_shape[:-1], self.out_features)
        return output

    @torch.no_grad()
    def update_grid(self, x: torch.Tensor, margin=0.01):
        assert x.dim() == 2 and x.size(1) == self.in_features
        batch = x.size(0)

        splines = self.b_splines(x)  # (batch, in, coeff)
        splines = splines.permute(1, 0, 2)  # (in, batch, coeff)
        orig_coeff = self.scaled_spline_weight  # (out, in, coeff)
        orig_coeff = orig_coeff.permute(1, 2, 0)  # (in, coeff, out)
        unreduced_spline_output = torch.bmm(splines, orig_coeff)  # (in, batch, out)
        unreduced_spline_output = unreduced_spline_output.permute(
            1, 0, 2
        )  # (batch, in, out)

        # sort each channel individually to collect data distribution
        x_sorted = torch.sort(x, dim=0)[0]
        grid_adaptive = x_sorted[
            torch.linspace(
                0, batch - 1, self.grid_size + 1, dtype=torch.int64, device=x.device
            )
        ]

        uniform_step = (x_sorted[-1] - x_sorted[0] + 2 * margin) / self.grid_size
        grid_uniform = (
                torch.arange(
                    self.grid_size + 1, dtype=torch.float32, device=x.device
                ).unsqueeze(1)
                * uniform_step
                + x_sorted[0]
                - margin
        )

        grid = self.grid_eps * grid_uniform + (1 - self.grid_eps) * grid_adaptive
        grid = torch.concatenate(
            [
                grid[:1]
                - uniform_step
                * torch.arange(self.spline_order, 0, -1, device=x.device).unsqueeze(1),
                grid,
                grid[-1:]
                + uniform_step
                * torch.arange(1, self.spline_order + 1, device=x.device).unsqueeze(1),
            ],
            dim=0,
        )

        self.grid.copy_(grid.T)
        self.spline_weight.data.copy_(self.curve2coeff(x, unreduced_spline_output))

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        """
        Compute the regularization loss.

        This is a dumb simulation of the original L1 regularization as stated in the
        paper, since the original one requires computing absolutes and entropy from the
        expanded (batch, in_features, out_features) intermediate tensor, which is hidden
        behind the F.linear function if we want an memory efficient implementation.

        The L1 regularization is now computed as mean absolute value of the spline
        weights. The authors implementation also includes this term in addition to the
        sample-based regularization.
        """
        l1_fake = self.spline_weight.abs().mean(-1)
        regularization_loss_activation = l1_fake.sum()
        p = l1_fake / regularization_loss_activation
        regularization_loss_entropy = -torch.sum(p * p.log())
        return (
                regularize_activation * regularization_loss_activation
                + regularize_entropy * regularization_loss_entropy
        )




