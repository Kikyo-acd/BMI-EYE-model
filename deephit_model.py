import torch
import torch.nn as nn
import torch.nn.functional as F

class DeepHit(nn.Module):
    """
    DeepHit模型。
    将生存时间离散化为多个区间，预测在每个区间的事件发生概率。
    损失函数由似然损失和排序损失两部分构成。

    参数:
        input_dim (int): 输入特征的数量。
        num_nodes (int): 隐藏层的基础节点数。
        k_bins (int): 生存时间离散化的区间数量。
        alpha (float): 排序损失在总损失中的权重。
        sigma (float): 排序损失中的平滑参数。
        weight_decay (float): L2正则化系数。
    """
    def __init__(self, input_dim, num_nodes=64, k_bins=10, alpha=0.5, sigma=0.1, weight_decay=1e-4):
        super(DeepHit, self).__init__()
        self.k_bins = k_bins
        self.alpha = alpha
        self.sigma = sigma
        self.weight_decay = weight_decay
        
        self.shared_layer = nn.Sequential(
            nn.Linear(input_dim, num_nodes * 2),
            nn.BatchNorm1d(num_nodes * 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(num_nodes * 2, num_nodes),
            nn.BatchNorm1d(num_nodes),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        self.output_layer = nn.Linear(num_nodes, k_bins)

    def forward(self, x):
        """前向传播，返回在每个时间区间的事件概率。"""
        shared_features = self.shared_layer(x)
        out = self.output_layer(shared_features)
        return F.softmax(out, dim=1) # Softmax保证概率和为1

    def loss(self, probs, times, events, time_bins):
        """
        DeepHit损失函数，包含负对数似然损失和排序损失。
        使用向量化实现以提高效率。
        """
        device = probs.device
        batch_size = probs.shape[0]
        
        # 将连续时间映射到离散区间索引
        time_idx = torch.bucketize(times, time_bins[1:-1])

        # 1. 似然损失 (仅对发生事件的样本)
        event_mask = (events == 1)
        if event_mask.sum() > 0:
            # 提取发生事件的样本，在它们对应事件区间的预测概率
            log_likelihood = torch.log(probs[event_mask, time_idx[event_mask]] + 1e-8)
            nll_loss = -log_likelihood.mean()
        else:
            # 如果批次中无事件，则似然损失为0，但需保持计算图连接
            nll_loss = 0.0 * probs.sum()

        # 2. 排序损失 (向量化实现)
        time_diff = times.view(-1, 1) - times.view(1, -1) # T_i - T_j
        # 找出 E_i=1, E_j=0 的样本对
        event_pairs = events.view(-1, 1) * (1 - events.view(1, -1)) 
        # 有效对：E_i=1, E_j=0 且 T_i < T_j
        valid_pairs = (event_pairs == 1) & (time_diff < 0)
        
        if valid_pairs.sum() > 0:
            cum_probs = torch.cumsum(probs, dim=1)
            # 提取每个样本在其事件时间的累积风险
            risk_i = cum_probs.gather(1, time_idx.view(-1, 1)).squeeze()
            
            eta_i = risk_i.view(-1, 1)
            eta_j = risk_i.view(1, -1)
            rank_loss_matrix = torch.exp(-(eta_i - eta_j) / self.sigma)
            # 只对有效的排序对计算损失并取平均
            rank_loss = (rank_loss_matrix * valid_pairs.float()).sum() / valid_pairs.float().sum()
        else:
            # 如果批次中无可排序的对，则排序损失为0，但需保持计算图连接
            rank_loss = 0.0 * probs.sum()

        # L2 正则化
        l2_reg = 0
        for param in self.parameters():
            l2_reg += torch.sum(param ** 2)

        return nll_loss + self.alpha * rank_loss + self.weight_decay * l2_reg / 2

    @torch.no_grad()
    def predict_risk(self, x):
        """预测累积风险 F(t) = P(T <= t)"""
        self.eval()
        probs = self.forward(x)
        return torch.cumsum(probs, dim=1)

    @torch.no_grad()
    def predict_survival(self, x):
        """预测生存概率 S(t) = 1 - F(t)"""
        self.eval()
        cum_risk = self.predict_risk(x)
        return 1 - cum_risk
