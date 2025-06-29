import pandas as pd
import numpy as np
from lifelines import CoxPHFitter
from scipy.interpolate import interp1d

class CoxModel:
    """
    A wrapper for the Cox Proportional Hazards model.
    This class uses lifelines.CoxPHFitter as its core.
    
    Args:
        penalizer (float): The penalizer (regularization) coefficient for the Cox model.
    """
    def __init__(self, penalizer=0.1):
        self.model = CoxPHFitter(penalizer=penalizer)

    def fit(self, X, T, E):
        """
        Fits the Cox model.
        
        Args:
            X (np.array): The feature matrix.
            T (np.array): The event/censoring times.
            E (np.array): The event indicator (1=event, 0=censored).
        """
        # lifelines requires data in a pandas.DataFrame format
        df = pd.DataFrame(X, columns=[f'f_{i}' for i in range(X.shape[1])])
        df['T'] = T
        df['E'] = E
        self.model.fit(df, duration_col='T', event_col='E')
        return self

    def predict_risk(self, X):
        """
        Predicts the risk score.
        lifelines' predict_partial_hazard returns scores where higher means better survival
        (lower risk). We negate it to unify the convention that "higher score = higher risk".
        """
        df = pd.DataFrame(X, columns=[f'f_{i}'for i in range(X.shape[1])])
        return -self.model.predict_partial_hazard(df).values

    def predict_survival(self, X, times):
        """
        Predicts the survival probability at specified time points.
        """
        df = pd.DataFrame(X, columns=[f'f_{i}' for i in range(X.shape[1])])
        # Predict the survival function for each sample
        surv_funcs = self.model.predict_survival_function(df)
        
        survival_probs = np.zeros((len(X), len(times)))
        # Interpolate for each sample
        for i in range(len(surv_funcs.columns)):
            surv_func = surv_funcs.iloc[:, i] # Get the survival function (a Series) for a single sample
            interp = interp1d(
                surv_func.index, surv_func.values, kind='previous', 
                bounds_error=False, fill_value=(1.0, surv_func.values[-1])
            )
            survival_probs[i, :] = interp(times)
            
        return survival_probs.T # Return shape [n_samples, n_times]
