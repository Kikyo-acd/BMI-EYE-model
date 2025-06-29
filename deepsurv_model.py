# deepsurv_model.py
# Contains the definition for the DeepSurv model, its loss function, and prediction methods.

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

class DeepSurv(nn.Module):
    """
    DeepSurv Model.
    This is a deep learning-based Cox Proportional Hazards model. It uses a neural
    network to predict the log-risk ratio for an individual.
    
    Args:
        input_dim (int): The number of input features.
        hidden_dims (list): A list containing the number of nodes in each hidden layer.
        dropout (float): The dropout rate.
        weight_decay (float): The coefficient for L2 regularization (weight decay).
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
        layers.append(nn.Linear(prev_dim, 1)) # Output layer: a single node for the log-risk score
        
        self.model = nn.Sequential(*layers)
        self.weight_decay = weight_decay
        
        # Attributes for storing the estimated baseline hazard
        self.baseline_hazard_ = None
        self.cumulative_baseline_hazard_ = None
        self.time_points_ = None

    def forward(self, x):
        """Forward pass, returns the log-risk scores."""
        return self.model(x)

    def loss(self, risk_scores, times, events):
        """
        Calculates the negative log Cox partial likelihood loss.
        This is the core loss function for DeepSurv.
        """
        risk_scores = risk_scores.squeeze()
        
        # Sort by event time in descending order, a prerequisite for calculating the risk set
        sorted_indices = torch.argsort(times, descending=True)
        risk_scores_sorted = risk_scores[sorted_indices]
        events_sorted = events[sorted_indices]
        
        # Calculate the hazard ratio, e^h(x)
        hazard_ratio = torch.exp(risk_scores_sorted)
        
        # Calculate the log-risk-sum for the risk set at each time point
        log_risk = torch.log(torch.cumsum(hazard_ratio, dim=0))
        
        # Calculate the partial likelihood for each individual who experienced an event
        # The loss is only calculated for uncensored samples (events_sorted == 1)
        uncensored_likelihood = risk_scores_sorted - log_risk
        censored_likelihood = uncensored_likelihood * events_sorted
        
        num_events = torch.sum(events_sorted)
        # Average the loss and take its negative
        neg_likelihood = -torch.sum(censored_likelihood) / (num_events + 1e-8)
        
        # Add L2 regularization term
        l2_reg = 0
        for param in self.parameters():
            l2_reg += torch.sum(param ** 2)
        
        return neg_likelihood + self.weight_decay * l2_reg / 2

    def fit_baseline_hazard(self, X_train, T_train, E_train):
        """
        After model training, this method uses the Breslow estimator to estimate
        the baseline hazard function, which is necessary for predicting absolute
        survival probabilities.
        """
        print("Estimating baseline hazard for DeepSurv...")
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
        print(f"Baseline hazard estimation complete, covering {len(self.time_points_)} unique event time points.")
        return self

    def predict_survival(self, x, times):
        """
        Predicts the survival probability S(t|x) at specified time points.
        Formula: S(t|x) = exp(-H0(t) * exp(h(x)))
        """
        if self.cumulative_baseline_hazard_ is None:
            raise RuntimeError("The `fit_baseline_hazard` method must be called first to estimate the baseline hazard.")
            
        device = next(self.parameters()).device
        if isinstance(x, np.ndarray):
            x_tensor = torch.FloatTensor(x).to(device)
        else:
            x_tensor = x.to(device)

        with torch.no_grad():
            self.eval()
            risk_scores = self(x_tensor).squeeze().cpu().numpy()
            
        # Use interpolation (in a step-function manner) to get the cumulative baseline hazard at specified times
        f = interp1d(
            self.time_points_,
            self.cumulative_baseline_hazard_.values.squeeze(),
            kind='previous', # 'previous' implements the behavior of a step function
            bounds_error=False,
            fill_value=(0, self.cumulative_baseline_hazard_.values[-1])
        )
        H0_t = f(times) # Cumulative baseline hazard H0(t)
        
        # Calculate survival probabilities
        survival_probs = np.exp(-np.outer(np.exp(risk_scores), H0_t))
        return survival_probs.T # Return shape [n_samples, n_times]
