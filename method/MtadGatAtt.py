import torch
import torch.nn as nn
from math import sqrt
import torch.nn.functional as F
import numpy as np
import torch.utils.data as tud


class ConvLayer(nn.Module):
    """1-D Convolution layer to extract high-level features of each time-series input
    :param n_features: Number of input features/nodes
    :param window_size: length of the input sequence
    :param kernel_size: size of kernel to use in the convolution operation
    """

    def __init__(self, n_features, kernel_size=7):
        super(ConvLayer, self).__init__()
        self.padding = nn.ConstantPad1d((kernel_size - 1) // 2, 0.0)
        self.conv = nn.Conv1d(in_channels=n_features, out_channels=n_features, kernel_size=kernel_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.padding(x)
        x = self.relu(self.conv(x))
        return x.permute(0, 2, 1)  # Permute back


class FeatureAttentionLayer(nn.Module):
    """Single Graph Feature/Spatial Attention Layer
    :param n_features: Number of input features/nodes
    :param window_size: length of the input sequence
    :param dropout: percentage of nodes to dropout
    :param alpha: negative slope used in the leaky rely activation function
    :param embed_dim: embedding dimension (output dimension of linear transformation)
    :param use_gatv2: whether to use the modified attention mechanism of GATv2 instead of standard GAT
    :param use_bias: whether to include a bias term in the attention layer
    """

    def __init__(self, n_features, window_size, dropout, alpha, embed_dim=None, use_gatv2=True, use_bias=True,
                 use_softmax=True):
        super(FeatureAttentionLayer, self).__init__()
        self.n_features = n_features
        self.window_size = window_size
        self.dropout = dropout
        self.embed_dim = embed_dim if embed_dim is not None else window_size
        self.use_gatv2 = use_gatv2
        self.num_nodes = n_features
        self.use_bias = use_bias
        self.use_softmax = use_softmax

        # Because linear transformation is done after concatenation in GATv2
        if self.use_gatv2:
            self.embed_dim *= 2
            lin_input_dim = 2 * window_size
            a_input_dim = self.embed_dim
        else:
            lin_input_dim = window_size
            a_input_dim = 2 * self.embed_dim

        self.lin = nn.Linear(lin_input_dim, self.embed_dim)
        self.a = nn.Parameter(torch.empty((a_input_dim, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        if self.use_bias:
            self.bias = nn.Parameter(torch.ones(n_features, n_features))

        self.leakyrelu = nn.LeakyReLU(alpha)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x shape (b, n, k): b - batch size, n - window size, k - number of features
        # For feature attention we represent a node as the values of a particular feature across all timestamps

        x = x.permute(0, 2, 1)

        # 'Dynamic' GAT attention
        # Proposed by Brody et. al., 2021 (https://arxiv.org/pdf/2105.14491.pdf)
        # Linear transformation applied after concatenation and attention layer applied after leakyrelu
        if self.use_gatv2:
            a_input = self._make_attention_input(x)                 # (b, k, k, 2*window_size)
            a_input = self.leakyrelu(self.lin(a_input))             # (b, k, k, embed_dim)
            e = torch.matmul(a_input, self.a).squeeze(3)            # (b, k, k, 1)

        # Original GAT attention
        else:
            Wx = self.lin(x)                                                  # (b, k, k, embed_dim)
            a_input = self._make_attention_input(Wx)                          # (b, k, k, 2*embed_dim)
            e = self.leakyrelu(torch.matmul(a_input, self.a)).squeeze(3)      # (b, k, k, 1)

        if self.use_bias:
            e += self.bias

        # Attention weights
        if self.use_softmax:
            e = torch.softmax(e, dim=2)
        attention = torch.dropout(e, self.dropout, train=self.training)

        # Computing new node features using the attention
        h = self.sigmoid(torch.matmul(attention, x))

        return h.permute(0, 2, 1)

    def _make_attention_input(self, v):
        """Preparing the feature attention mechanism.
        Creating matrix with all possible combinations of concatenations of node.
        Each node consists of all values of that node within the window
            v1 || v1,
            ...
            v1 || vK,
            v2 || v1,
            ...
            v2 || vK,
            ...
            ...
            vK || v1,
            ...
            vK || vK,
        """

        K = self.num_nodes
        blocks_repeating = v.repeat_interleave(K, dim=1)  # Left-side of the matrix
        blocks_alternating = v.repeat(1, K, 1)  # Right-side of the matrix
        combined = torch.cat((blocks_repeating, blocks_alternating), dim=2)  # (b, K*K, 2*window_size)

        if self.use_gatv2:
            return combined.view(v.size(0), K, K, 2 * self.window_size)
        else:
            return combined.view(v.size(0), K, K, 2 * self.embed_dim)


class TemporalAttentionLayer(nn.Module):
    """Single Graph Temporal Attention Layer
    :param n_features: number of input features/nodes
    :param window_size: length of the input sequence
    :param dropout: percentage of nodes to dropout
    :param alpha: negative slope used in the leaky rely activation function
    :param embed_dim: embedding dimension (output dimension of linear transformation)
    :param use_gatv2: whether to use the modified attention mechanism of GATv2 instead of standard GAT
    :param use_bias: whether to include a bias term in the attention layer

    """

    def __init__(self, n_features, window_size, dropout, alpha, embed_dim=None, use_gatv2=True, use_bias=True,
                 use_softmax=True):
        super(TemporalAttentionLayer, self).__init__()
        self.n_features = n_features
        self.window_size = window_size
        self.dropout = dropout
        self.use_gatv2 = use_gatv2
        self.embed_dim = embed_dim if embed_dim is not None else n_features
        self.num_nodes = window_size
        self.use_bias = use_bias
        self.use_softmax = use_softmax

        # Because linear transformation is performed after concatenation in GATv2
        if self.use_gatv2:
            self.embed_dim *= 2
            lin_input_dim = 2 * n_features
            a_input_dim = self.embed_dim
        else:
            lin_input_dim = n_features
            a_input_dim = 2 * self.embed_dim

        self.lin = nn.Linear(lin_input_dim, self.embed_dim)
        self.a = nn.Parameter(torch.empty((a_input_dim, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        if self.use_bias:
            self.bias = nn.Parameter(torch.ones(window_size, window_size))

        self.leakyrelu = nn.LeakyReLU(alpha)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x shape (b, n, k): b - batch size, n - window size, k - number of features
        # For temporal attention a node is represented as all feature values at a specific timestamp

        # 'Dynamic' GAT attention
        # Proposed by Brody et. al., 2021 (https://arxiv.org/pdf/2105.14491.pdf)
        # Linear transformation applied after concatenation and attention layer applied after leakyrelu
        if self.use_gatv2:
            a_input = self._make_attention_input(x)              # (b, n, n, 2*n_features)
            a_input = self.leakyrelu(self.lin(a_input))          # (b, n, n, embed_dim)
            e = torch.matmul(a_input, self.a).squeeze(3)         # (b, n, n, 1)

        # Original GAT attention
        else:
            Wx = self.lin(x)                                                  # (b, n, n, embed_dim)
            a_input = self._make_attention_input(Wx)                          # (b, n, n, 2*embed_dim)
            e = self.leakyrelu(torch.matmul(a_input, self.a)).squeeze(3)      # (b, n, n, 1)

        if self.use_bias:
            e += self.bias  # (b, n, n, 1)

        # Attention weights
        if self.use_softmax:
            e = torch.softmax(e, dim=2)
        attention = torch.dropout(e, self.dropout, train=self.training)

        h = self.sigmoid(torch.matmul(attention, x))    # (b, n, k)

        return h

    def _make_attention_input(self, v):
        """Preparing the temporal attention mechanism.
        Creating matrix with all possible combinations of concatenations of node values:
            (v1, v2..)_t1 || (v1, v2..)_t1
            (v1, v2..)_t1 || (v1, v2..)_t2

            ...
            ...

            (v1, v2..)_tn || (v1, v2..)_t1
            (v1, v2..)_tn || (v1, v2..)_t2

        """

        K = self.num_nodes
        blocks_repeating = v.repeat_interleave(K, dim=1)  # Left-side of the matrix
        blocks_alternating = v.repeat(1, K, 1)  # Right-side of the matrix
        combined = torch.cat((blocks_repeating, blocks_alternating), dim=2)

        if self.use_gatv2:
            return combined.view(v.size(0), K, K, 2 * self.n_features)
        else:
            return combined.view(v.size(0), K, K, 2 * self.embed_dim)


class FullAttention(nn.Module):
    def __init__(self, mask_flag=True, scale=None, attention_dropout=0.1, output_attention=False):
        super(FullAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)
        self.relu_q = nn.ReLU()
        self.relu_k = nn.ReLU()

    @staticmethod
    def TriangularCausalMask(B, L, S, device='cpu'):
        mask_shape = [B, 1, L, S]
        with torch.no_grad():
            mask = torch.triu(torch.ones(mask_shape, dtype=torch.bool), diagonal=1)
            return mask.to(device)

    def forward(self, queries, keys, values, attn_mask):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1. / sqrt(E) # scale相对于取多少比例，取前1/根号n

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)
        if self.mask_flag:
            if attn_mask is None:
                attn_mask = self.TriangularCausalMask(B, L, S, device=queries.device)

            scores.masked_fill_(attn_mask, 0)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)

        # queries = self.relu_q(queries)
        # keys = self.relu_k(keys)
        # KV = torch.einsum("blhe,bshe->bhls", keys, values)
        # A = self.dropout(scale * KV)
        # V = torch.einsum("bshd,bhls->blhd", queries, A)

        if self.output_attention:
            return (V.contiguous(), A)
        else:
            return (V.contiguous(), None)


class ProbAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=2, scale=None, attention_dropout=0.1, output_attention=False):
        super(ProbAttention, self).__init__()
        self.factor = factor
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention

    @staticmethod
    def ProbMask(B, H, D, index, scores, device='cpu'):
        _mask = torch.ones(D, scores.shape[-2], dtype=torch.bool).triu(1)
        _mask_ex = _mask[None, None, :].expand(B, H, D, scores.shape[-2])
        indicator = _mask_ex.transpose(-2, -1)[torch.arange(B)[:, None, None],
                    torch.arange(H)[None, :, None],
                    index, :].transpose(-2, -1)
        mask = indicator.view(scores.shape)
        return mask.to(device)

    def _prob_KV(self, K, V, sample_v, n_top):  # n_top: c*ln(L_q)
        # Q [B, H, L, D]
        B, H, L, E_V = V.shape
        _, _, _, E_K = K.shape

        # calculate the sampled K_V

        V_expand = V.transpose(-2, -1).unsqueeze(-2).expand(B, H, E_V, E_K, L)
        index_sample = torch.randint(E_V, (E_K, sample_v))  # real U = U_part(factor*ln(L_k))*L_q
        V_sample = V_expand[:, :, torch.arange(E_V).unsqueeze(1), index_sample, :]
        K_V_sample = torch.matmul(K.transpose(-2, -1).unsqueeze(-2), V_sample.transpose(-2, -1)).squeeze()

        # find the Top_k query with sparisty measurement
        M = K_V_sample.max(-1)[0] - torch.div(K_V_sample.sum(-1), E_V)
        M_top = M.topk(n_top, sorted=False)[1]

        # use the reduced Q to calculate Q_K
        V_reduce = V.transpose(-2, -1)[torch.arange(B)[:, None, None],
                   torch.arange(H)[None, :, None],
                   M_top, :].transpose(-2, -1)  # factor*ln(L_q)
        K_V = torch.matmul(K.transpose(-2, -1), V_reduce)  # factor*ln(L_q)*L_k
        #
        return K_V, M_top

    def _get_initial_context(self, V, L_Q):
        B, H, L_V, D = V.shape
        if not self.mask_flag:
            # V_sum = V.sum(dim=-2)
            V_sum = V.mean(dim=-2)
            contex = V_sum.unsqueeze(-2).expand(B, H, L_Q, V_sum.shape[-1]).clone()
        else:  # use mask
            assert (L_Q == L_V)  # requires that L_Q == L_V, i.e. for self-attention only
            contex = V.cumsum(dim=-2)
        return contex

    def _update_context(self, context_in, Q, scores, index, D_K, attn_mask):
        B, H, L, D_Q = Q.shape

        if self.mask_flag:
            attn_mask = self.ProbMask(B, H, D_K, index, scores, device=Q.device)
            scores.masked_fill_(attn_mask, -np.inf)

        attn = torch.softmax(scores, dim=-1)  # nn.Softmax(dim=-1)(scores)

        context_in.transpose(-2, -1)[torch.arange(B)[:, None, None],
        torch.arange(H)[None, :, None],
        index, :] = torch.matmul(Q, attn).type_as(context_in).transpose(-2, -1)
        if self.output_attention:
            attns = (torch.ones([B, H, D_K, D_K]) / D_K).type_as(attn)
            attns[torch.arange(B)[:, None, None], torch.arange(H)[None, :, None], index, :] = attn
            return (context_in, attns)
        else:
            return (context_in, None)

    def forward(self, queries, keys, values, attn_mask):
        # B, L_Q, H, D = queries.shape
        # _, L_K, _, _ = keys.shape

        B, L, H, D_K = keys.shape
        _, _, _, D_V = values.shape

        queries = queries.transpose(2, 1)
        keys = keys.transpose(2, 1)
        values = values.transpose(2, 1)

        U_part = self.factor * np.ceil(np.log(D_V)).astype('int').item()  # c*ln(L_k)
        u = self.factor * np.ceil(np.log(D_K)).astype('int').item()  # c*ln(L_q)

        U_part = U_part if U_part < D_V else D_V
        u = u if u < D_K else D_K

        scores_top, index = self._prob_KV(keys, values, sample_v=U_part, n_top=u)

        # add scale factor
        scale = self.scale or 1. / sqrt(D_K)
        if scale is not None:
            scores_top = scores_top * scale
        # get the context
        context = self._get_initial_context(queries, L)
        # update the context with selected top_k queries
        context, attn = self._update_context(context, queries, scores_top, index, D_K, attn_mask)

        return context.contiguous(), attn


class AttentionBlock(nn.Module):
    def __init__(self, d_model, n_model, n_heads=8, d_keys=None, d_values=None):
        super(AttentionBlock, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)
        self.inner_attention = FullAttention()
        # self.inner_attention = ProbAttention(device=device)
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, queries, keys, values, attn_mask):
        '''
        Q: [batch_size, len_q, d_k]
        K: [batch_size, len_k, d_k]
        V: [batch_size, len_v(=len_k), d_v]
        attn_mask: [batch_size, seq_len, seq_len]
        '''
        batch_size, len_q, _ = queries.shape
        _, len_k, _ = keys.shape

        queries = self.query_projection(queries).view(batch_size, len_q, self.n_heads, -1)
        keys = self.key_projection(keys).view(batch_size, len_k, self.n_heads, -1)
        values = self.value_projection(values).view(batch_size, len_k, self.n_heads, -1)

        out, attn = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask
        )
        out = out.view(batch_size, len_q, -1)
        out = self.out_projection(out)
        out = self.layer_norm(out)
        return out, attn


class GRULayer(nn.Module):
    """Gated Recurrent Unit (GRU) Layer
    :param in_dim: number of input features
    :param hid_dim: hidden size of the GRU
    :param n_layers: number of layers in GRU
    :param dropout: dropout rate
    """

    def __init__(self, in_dim, hid_dim, n_layers, dropout):
        super(GRULayer, self).__init__()
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.dropout = 0.0 if n_layers == 1 else dropout
        self.gru = nn.GRU(in_dim, hid_dim, num_layers=n_layers, batch_first=True, dropout=self.dropout)

    def forward(self, x):
        out, h = self.gru(x)
        out, h = out[-1, :, :], h[-1, :, :]  # Extracting from last layer
        return out, h


class RNNDecoder(nn.Module):
    """GRU-based Decoder network that converts latent vector into output
    :param in_dim: number of input features
    :param n_layers: number of layers in RNN
    :param hid_dim: hidden size of the RNN
    :param dropout: dropout rate
    """

    def __init__(self, in_dim, hid_dim, n_layers, dropout):
        super(RNNDecoder, self).__init__()
        self.in_dim = in_dim
        self.dropout = 0.0 if n_layers == 1 else dropout
        self.rnn = nn.GRU(in_dim, hid_dim, n_layers, batch_first=True, dropout=self.dropout)

    def forward(self, x):
        decoder_out, _ = self.rnn(x)
        return decoder_out


class ReconstructionModel(nn.Module):
    """Reconstruction Model
    :param window_size: length of the input sequence
    :param in_dim: number of input features
    :param n_layers: number of layers in RNN
    :param hid_dim: hidden size of the RNN
    :param in_dim: number of output features
    :param dropout: dropout rate
    """

    def __init__(self, window_size, in_dim, hid_dim, out_dim, n_layers, dropout):
        super(ReconstructionModel, self).__init__()
        self.window_size = window_size
        self.decoder = RNNDecoder(in_dim, hid_dim, n_layers, dropout)
        self.fc = nn.Linear(hid_dim, out_dim)

    def forward(self, x):
        # x will be last hidden state of the GRU layer
        h_end = x
        h_end_rep = h_end.repeat_interleave(self.window_size, dim=1).view(x.size(0), self.window_size, -1)

        decoder_out = self.decoder(h_end_rep)
        out = self.fc(decoder_out)
        return out


class Forecasting_Model(nn.Module):
    """Forecasting model (fully-connected network)
    :param in_dim: number of input features
    :param hid_dim: hidden size of the FC network
    :param out_dim: number of output features
    :param n_layers: number of FC layers
    :param dropout: dropout rate
    """

    def __init__(self, in_dim, hid_dim, out_dim, n_layers, dropout):
        super(Forecasting_Model, self).__init__()
        layers = [nn.Linear(in_dim, hid_dim)]
        for _ in range(n_layers - 1):
            layers.append(nn.Linear(hid_dim, hid_dim))

        layers.append(nn.Linear(hid_dim, out_dim))

        self.layers = nn.ModuleList(layers)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x):
        for i in range(len(self.layers) - 1):
            x = self.relu(self.layers[i](x))
            x = self.dropout(x)
        return self.layers[-1](x)


class Model(nn.Module):
    """ MTAD_GAT model class.

    :param n_features: Number of input features
    :param window_size: Length of the input sequence
    :param out_dim: Number of features to output
    :param kernel_size: size of kernel to use in the 1-D convolution
    :param feat_gat_embed_dim: embedding dimension (output dimension of linear transformation)
           in feat-oriented GAT layer
    :param time_gat_embed_dim: embedding dimension (output dimension of linear transformation)
           in time-oriented GAT layer
    :param use_gatv2: whether to use the modified attention mechanism of GATv2 instead of standard GAT
    :param gru_n_layers: number of layers in the GRU layer
    :param gru_hid_dim: hidden dimension in the GRU layer
    :param forecast_n_layers: number of layers in the FC-based Forecasting Model
    :param forecast_hid_dim: hidden dimension in the FC-based Forecasting Model
    :param recon_n_layers: number of layers in the GRU-based Reconstruction Model
    :param recon_hid_dim: hidden dimension in the GRU-based Reconstruction Model
    :param dropout: dropout rate
    :param alpha: negative slope used in the leaky rely activation function

    """

    def __init__(self, customs: dict, dataloader: tud.DataLoader = None):
        super(Model, self).__init__()
        n_features = dataloader.dataset.train_inputs.shape[-1]
        window_size = int(customs["input_size"])
        out_dim = n_features
        kernel_size = 7
        feat_gat_embed_dim = None
        time_gat_embed_dim = None
        use_gatv2 = True
        gru_n_layers = 1
        gru_hid_dim = 150
        forecast_n_layers = 1
        forecast_hid_dim = 150
        recon_n_layers = 1
        recon_hid_dim = 150
        dropout = 0.2
        alpha = 0.2
        optimize = True

        self.name = "MtadGatAtt"
        self.optimize = optimize
        use_softmax = not optimize

        self.conv = ConvLayer(n_features, kernel_size)
        self.feature_gat = FeatureAttentionLayer(
            n_features, window_size, dropout, alpha, feat_gat_embed_dim, use_gatv2, use_softmax=use_softmax)
        self.temporal_gat = TemporalAttentionLayer(n_features, window_size, dropout, alpha, time_gat_embed_dim,
                                                   use_gatv2, use_softmax=use_softmax)
        self.forecasting_model = Forecasting_Model(
            gru_hid_dim, forecast_hid_dim, out_dim, forecast_n_layers, dropout)
        if optimize:
            self.encode = AttentionBlock(3 * n_features, window_size)
            self.encode_feature = nn.Linear(3 * n_features * window_size, gru_hid_dim)
            self.decode_feature = nn.Linear(gru_hid_dim, n_features * window_size)
            self.decode = AttentionBlock(n_features, window_size)
        else:
            self.gru = GRULayer(3 * n_features, gru_hid_dim, gru_n_layers, dropout)
            self.recon_model = ReconstructionModel(window_size, gru_hid_dim, recon_hid_dim, out_dim, recon_n_layers,
                                                   dropout)

    def forward(self, x):
        x = self.conv(x)
        h_feat = self.feature_gat(x)
        h_temp = self.temporal_gat(x)
        h_cat = torch.cat([x, h_feat, h_temp], dim=2)  # (b, n, 3k)

        if self.optimize:
            h_end, _ = self.encode(h_cat, h_cat, h_cat, None)
            h_end = self.encode_feature(h_end.reshape(h_end.size(0), -1))
        else:
            _, h_end = self.gru(h_cat)
            h_end = h_end.view(x.shape[0], -1)  # Hidden state for last timestamp

        predictions = self.forecasting_model(h_end)

        if self.optimize:
            h_end = self.decode_feature(h_end)
            h_end = h_end.reshape(x.shape[0], x.shape[1], x.shape[2])
            recons, _ = self.decode(h_end, h_end, h_end, None)
        else:
            recons = self.recon_model(h_end)

        return predictions, recons

    def loss(self, x, y_true, epoch: int = None, device: str = "cpu"):
        preds, recons = self.forward(x)

        if preds.ndim == 3:
            preds = preds.squeeze(1)
        if y_true.ndim == 3:
            y_true = y_true.squeeze(1)
        forecast_criterion = nn.MSELoss()
        recon_criterion = nn.MSELoss()

        forecast_loss = torch.sqrt(forecast_criterion(y_true, preds))
        recon_loss = torch.sqrt(recon_criterion(x, recons))

        loss = forecast_loss + recon_loss
        loss.backward()
        return loss.item()

    def detection(self, x, y_true, epoch: int = None, device: str = "cpu"):
        preds, recons = self.forward(x)
        score = F.pairwise_distance(recons.reshape(recons.size(0), -1), x.reshape(x.size(0), -1)) + F.pairwise_distance(y_true.reshape(y_true.size(0), -1), preds.reshape(preds.size(0), -1))
        return score, None


if __name__ == "__main__":
    from tqdm import tqdm
    import time
    epoch = 10000
    batch_size = 1
    # device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
    device = 'cpu'
    input_len_list = [30, 60, 90, 120, 150, 180, 210, 240, 270, 300]
    for input_len in input_len_list:
        model = Model(52, input_len, 52, optimize=False, device=device).to(device)
        a = torch.Tensor(torch.ones((batch_size, input_len, 52))).to(device)
        start = time.time()
        for i in tqdm(range(epoch)):
            model(a)
        end = time.time()
        speed1 = batch_size * epoch / (end - start)

        model = Model(52, input_len, 52, optimize=True, device=device).to(device)
        a = torch.Tensor(torch.ones((batch_size, input_len, 52))).to(device)
        start = time.time()
        for i in tqdm(range(epoch)):
            model(a)
        end = time.time()
        speed2 = batch_size * epoch / (end - start)
        print(input_len, (speed2 - speed1)/speed1, speed1, speed2)

