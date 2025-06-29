import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

class DeepSurv(nn.Module):
    """
    DeepSurv模型。
    这是一个基于深度学习的Cox比例风险模型。它使用神经网络来预测个体的对数风险比。
    
    参数:
        input_dim (int): 输入特征的数量。
        hidden_dims (list): 一个包含每个隐藏层节点数的列表。
        dropout (float): Dropout比率。
        weight_decay (float): L2正则化（权重衰减）的系数。
    """
    def __init__(self, input_dim, hidden_dims=[64, 32], dropout=0.2, weight_decay=1e-4):
        super(DeepSurv, self).__init__()
        
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, 1)) # 输出层，一个节点代表对数风险比
        
        self.model = nn.Sequential(*layers)
        self.weight_decay = weight_decay
        
        # 用于存储估计出的基线风险
        self.baseline_hazard_ = None
        self.cumulative_baseline_hazard_ = None
        self.time_points_ = None

    def forward(self, x):
        """前向传播，返回对数风险分数值。"""
        return self.model(x)

    def loss(self, risk_scores, times, events):
        """
        计算负对数Cox部分似然损失。
        这是DeepSurv的核心损失函数。
        """
        risk_scores = risk_scores.squeeze()
        
        # 按事件时间降序排序，这是计算风险集的前提
        sorted_indices = torch.argsort(times, descending=True)
        risk_scores_sorted = risk_scores[sorted_indices]
        events_sorted = events[sorted_indices]
        
        # 计算风险比 e^h(x)
        hazard_ratio = torch.exp(risk_scores_sorted)
        
        # 计算在每个时间点的风险集（所有“处于危险中”的个体）的对数风险和
        log_risk = torch.log(torch.cumsum(hazard_ratio, dim=0))
        
        # 计算每个发生事件个体的部分似然
        # 只对事件真实发生的样本（events_sorted == 1）计算损失
        uncensored_likelihood = risk_scores_sorted - log_risk
        censored_likelihood = uncensored_likelihood * events_sorted
        
        num_events = torch.sum(events_sorted)
        # 对损失进行平均，并取负
        neg_likelihood = -torch.sum(censored_likelihood) / (num_events + 1e-8)
        
        # 添加L2正则化项
        l2_reg = 0
        for param in self.parameters():
            l2_reg += torch.sum(param ** 2)
        
        return neg_likelihood + self.weight_decay * l2_reg / 2

    def fit_baseline_hazard(self, X_train, T_train, E_train):
        """
        在模型训练后，使用Breslow估计器来估计基线风险函数。
        这是预测绝对生存概率所必需的。
        """
        print("正在为DeepSurv估计基线风险...")
        device = next(self.parameters()).device
        X_tensor = torch.FloatTensor(X_train).to(device)

        with torch.no_grad():
            self.eval()
            risk_scores = self(X_tensor).squeeze().cpu().numpy()
        
        hazard_ratios = np.exp(risk_scores)
        
        unique_event_times = np.unique(T_train[E_train == 1])
        baseline_hazard = pd.DataFrame(
            np.zeros_like(unique_event_times, dtype=float),
            index=unique_event_times,
            columns=['baseline hazard']
        )

        for t in unique_event_times:
            events_at_t = (T_train == t) & (E_train == 1)
            at_risk = T_train >= t
            n_events = np.sum(events_at_t)
            risk_pool_sum = np.sum(hazard_ratios[at_risk])
            baseline_hazard.loc[t] = n_events / (risk_pool_sum + 1e-8)

        cumulative_baseline_hazard = baseline_hazard.cumsum()
        
        self.baseline_hazard_ = baseline_hazard
        self.cumulative_baseline_hazard_ = cumulative_baseline_hazard
        self.time_points_ = cumulative_baseline_hazard.index.values
        print(f"基线风险估计完成，覆盖 {len(self.time_points_)} 个独特事件时间点。")
        return self

    def predict_survival(self, x, times):
        """
        预测在指定时间点的生存概率 S(t|x)。
        公式: S(t|x) = exp(-H0(t) * exp(h(x)))
        """
        if self.cumulative_baseline_hazard_ is None:
            raise RuntimeError("必须先调用 `fit_baseline_hazard` 来估计基线风险。")
            
        device = next(self.parameters()).device
        if isinstance(x, np.ndarray):
            x_tensor = torch.FloatTensor(x).to(device)
        else:
            x_tensor = x.to(device)

        with torch.no_grad():
            self.eval()
            risk_scores = self(x_tensor).squeeze().cpu().numpy()
            
        # 使用线性插值（阶梯函数形式）来获取在指定时间点的累积基线风险
        f = interp1d(
            self.time_points_,
            self.cumulative_baseline_hazard_.values.squeeze(),
            kind='previous', # 'previous' 实现了阶梯函数的行为
            bounds_error=False,
            fill_value=(0, self.cumulative_baseline_hazard_.values[-1])
        )
        H0_t = f(times) # 累积基线风险 H0(t)
        
        # 计算生存概率
        survival_probs = np.exp(-np.outer(np.exp(risk_scores), H0_t))
        return survival_probs.T # 返回形状 [n_samples, n_times]
