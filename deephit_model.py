import torch
import torch.nn as nn
import torch.nn.functional as F

class DeepHit(nn.Module):
    """
    DeepHit Model.
    This model discretizes survival time into several intervals and predicts the
    probability of an event occurring in each interval. Its loss function is
    composed of a likelihood loss and a ranking loss component.

    Args:
        input_dim (int): The number of input features.
        num_nodes (int): The base number of nodes for hidden layers.
        k_bins (int): The number of discrete time intervals (bins).
        alpha (float): The weight of the ranking loss in the total loss.
        sigma (float): The smoothing parameter for the ranking loss.
        weight_decay (float): The coefficient for L2 regularization.
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
        """Forward pass, returns the event probabilities for each time interval."""
        shared_features = self.shared_layer(x)
        out = self.output_layer(shared_features)
        return F.softmax(out, dim=1) # Softmax ensures the probabilities sum to 1

    def loss(self, probs, times, events, time_bins):
        """
        DeepHit loss function, composed of negative log-likelihood and ranking loss.
        Implemented in a vectorized manner for efficiency.
        """
        device = probs.device
        batch_size = probs.shape[0]
        
        # Map continuous time to discrete interval indices
        time_idx = torch.bucketize(times, time_bins[1:-1])

        # 1. Likelihood loss (for uncensored samples only)
        event_mask = (events == 1)
        if event_mask.sum() > 0:
            # Select probabilities for samples that had an event, at their corresponding event interval
            log_likelihood = torch.log(probs[event_mask, time_idx[event_mask]] + 1e-8)
            nll_loss = -log_likelihood.mean()
        else:
            # If no events in the batch, likelihood loss is 0, but must remain connected to the graph
            nll_loss = 0.0 * probs.sum()

        # 2. Ranking loss (vectorized implementation)
        time_diff = times.view(-1, 1) - times.view(1, -1) # T_i - T_j
        # Find pairs where E_i=1 and E_j=0
        event_pairs = events.view(-1, 1) * (1 - events.view(1, -1)) 
        # Valid pairs for ranking: E_i=1, E_j=0, and T_i < T_j
        valid_pairs = (event_pairs == 1) & (time_diff < 0)
        
        if valid_pairs.sum() > 0:
            cum_probs = torch.cumsum(probs, dim=1)
            # Gather the cumulative risk for each sample at its event/censoring time
            risk_i = cum_probs.gather(1, time_idx.view(-1, 1)).squeeze()
            
            eta_i = risk_i.view(-1, 1)
            eta_j = risk_i.view(1, -1)
            rank_loss_matrix = torch.exp(-(eta_i - eta_j) / self.sigma)
            # Average the loss only over the valid pairs
            rank_loss = (rank_loss_matrix * valid_pairs.float()).sum() / valid_pairs.float().sum()
        else:
            # If no rankable pairs, ranking loss is 0, but must remain connected to the graph
            rank_loss = 0.0 * probs.sum()

        # L2 Regularization
        l2_reg = 0
        for param in self.parameters():
            l2_reg += torch.sum(param ** 2)

        return nll_loss + self.alpha * rank_loss + self.weight_decay * l2_reg / 2

    @torch.no_grad()
    def predict_risk(self, x):
        """Predicts the cumulative risk F(t) = P(T <= t)"""
        self.eval()
        probs = self.forward(x)
        return torch.cumsum(probs, dim=1)

    @torch.no_grad()
    def predict_survival(self, x):
        """Predicts the survival probability S(t) = 1 - F(t)"""
        self.eval()
        cum_risk = self.predict_risk(x)
        return 1 - cum_risk
