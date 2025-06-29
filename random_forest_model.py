import numpy as np
from sklearn.ensemble import RandomForestClassifier

class RandomForestSurvival:
    """
    A Random Forest-based survival model.
    The basic idea is to train an independent Random Forest classifier at each
    discrete time point to predict whether an event will have occurred by that time.
    
    Args:
        n_estimators (int): The number of trees in the forest.
        max_depth (int): The maximum depth of each tree.
        random_state (int): The random seed for reproducibility.
    """
    def __init__(self, n_estimators=100, max_depth=10, random_state=42):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        self.models = {}
        self.time_points = None

    def fit(self, X, T, E, time_points):
        """
        Fits the model.
        
        Args:
            X (np.array): The feature matrix.
            T (np.array): The event/censoring times.
            E (np.array): The event indicator.
            time_points (np.array): A series of discrete time points at which to
                                    train a classifier.
        """
        self.time_points = time_points
        for t in self.time_points:
            # Create the binary label: did an event happen by time t?
            y_t = ((T <= t) & (E == 1)).astype(int)
            
            rf = RandomForestClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                random_state=self.random_state,
                n_jobs=-1
            )
            rf.fit(X, y_t)
            self.models[t] = rf
        return self

    def predict_risk(self, X):
        """
        Predicts the event probability (risk) at each time point.
        Returns a matrix of shape [n_samples, n_time_points].
        """
        risk_probs = np.zeros((len(X), len(self.time_points)))
        for i, t in enumerate(self.time_points):
            # Predict the probability of class 1 (event)
            risk_probs[:, i] = self.models[t].predict_proba(X)[:, 1]
        return risk_probs

    def predict_survival(self, X):
        """
        Predicts the survival probability, S(t) = 1 - P(event by time t).
        """
        risk_probs = self.predict_risk(X)
        return 1 - risk_probs
