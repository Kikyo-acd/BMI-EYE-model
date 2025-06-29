import torch
import torch.nn as nn
import torch.nn.functional as F

class SurvivalTransformer(nn.Module):
    """
    基于Transformer的生存模型。
    使用Transformer编码器来学习特征间的复杂关系。
    其损失函数和预测逻辑与DeepHit类似，也是一个离散时间模型。

    参数:
        input_dim (int): 输入特征的数量。
        d_model (int): Transformer模型的内部维度。
        nhead (int): Transformer多头注意力机制的头数。
        num_layers (int): Transformer编码器的层数。
        dim_feedforward (int): Transformer前馈网络的维度。
        ... (其他参数与DeepHit类似)
    """
    def __init__(self, input_dim, d_model=64, nhead=4, num_layers=3, dim_feedforward=128,
                 dropout=0.2, k_bins=10, weight_decay=1e-4, alpha=0.5, sigma=0.1):
        super(SurvivalTransformer, self).__init__()

        self.k_bins = k_bins
        self.weight_decay = weight_decay
        self.alpha = alpha
        self.sigma = sigma

        self.embedding = nn.Linear(input_dim, d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, 
            dropout=dropout, batch_first=True, activation="gelu"
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, k_bins)
        )

    def forward(self, x):
        """前向传播。将静态数据视为长度为1的序列进行处理。"""
        # [batch, input_dim] -> [batch, d_model]
        x_embedded = self.embedding(x)
        # [batch, d_model] -> [batch, 1, d_model]
        x_seq = x_embedded.unsqueeze(1)
        
        # Transformer处理
        transformer_out = self.transformer_encoder(x_seq)
        
        # [batch, 1, d_model] -> [batch, d_model]
        out = transformer_out.squeeze(1)
        
        # 输出头
        logits = self.head(out)
        
        return F.softmax(logits, dim=1)

    def loss(self, probs, times, events, time_bins):
        """损失函数与DeepHit完全相同。"""
        device = probs.device
        batch_size = probs.shape[0]
        time_idx = torch.bucketize(times, time_bins[1:-1])
        event_mask = (events == 1)
        if event_mask.sum() > 0:
            log_likelihood = torch.log(probs[event_mask, time_idx[event_mask]] + 1e-8)
            nll_loss = -log_likelihood.mean()
        else:
            nll_loss = 0.0 * probs.sum()
        
        time_diff = times.view(-1, 1) - times.view(1, -1)
        event_pairs = events.view(-1, 1) * (1 - events.view(1, -1))
        valid_pairs = (event_pairs == 1) & (time_diff < 0)
        if valid_pairs.sum() > 0:
            cum_probs = torch.cumsum(probs, dim=1)
            risk_i = cum_probs.gather(1, time_idx.view(-1, 1)).squeeze()
            eta_i = risk_i.view(-1, 1)
            eta_j = risk_i.view(1, -1)
            rank_loss_matrix = torch.exp(-(eta_i - eta_j) / self.sigma)
            rank_loss = (rank_loss_matrix * valid_pairs.float()).sum() / valid_pairs.float().sum()
        else:
            rank_loss = 0.0 * probs.sum()

        l2_reg = 0
        for param in self.parameters():
            l2_reg += torch.sum(param ** 2)
            
        return nll_loss + self.alpha * rank_loss + self.weight_decay * l2_reg / 2

    @torch.no_grad()
    def predict_risk(self, x):
        self.eval()
        probs = self.forward(x)
        return torch.cumsum(probs, dim=1)

    @torch.no_grad()
    def predict_survival(self, x):
        self.eval()
        cum_risk = self.predict_risk(x)
        return 1 - cum_risk
