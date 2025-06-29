import pandas as pd
import numpy as np
from lifelines import CoxPHFitter
from scipy.interpolate import interp1d

class CoxModel:
    """
    Cox比例风险模型的包装器。
    使用 lifelines.CoxPHFitter 作为核心。
    
    参数:
        penalizer (float): Cox模型的惩罚项（正则化）系数。
    """
    def __init__(self, penalizer=0.1):
        self.model = CoxPHFitter(penalizer=penalizer)

    def fit(self, X, T, E):
        """
        拟合Cox模型。
        
        参数:
            X (np.array): 特征矩阵。
            T (np.array): 事件/删失时间。
            E (np.array): 事件指示器 (1=事件发生, 0=删失)。
        """
        # lifelines需要pandas.DataFrame格式的数据
        df = pd.DataFrame(X, columns=[f'f_{i}' for i in range(X.shape[1])])
        df['T'] = T
        df['E'] = E
        self.model.fit(df, duration_col='T', event_col='E')
        return self

    def predict_risk(self, X):
        """
        预测风险分数。
        lifelines的predict_partial_hazard分数越高表示生存越好（风险越低）。
        我们对其取负，使其统一为“分数越高，风险越高”。
        """
        df = pd.DataFrame(X, columns=[f'f_{i}'for i in range(X.shape[1])])
        return -self.model.predict_partial_hazard(df).values

    def predict_survival(self, X, times):
        """
        预测在指定时间点的生存概率。
        """
        df = pd.DataFrame(X, columns=[f'f_{i}' for i in range(X.shape[1])])
        # 预测每个样本的生存函数
        surv_funcs = self.model.predict_survival_function(df)
        
        survival_probs = np.zeros((len(X), len(times)))
        # 为每个样本进行插值
        for i in range(len(surv_funcs.columns)):
            surv_func = surv_funcs.iloc[:, i] # 提取单个样本的生存函数(Series)
            interp = interp1d(
                surv_func.index, surv_func.values, kind='previous', 
                bounds_error=False, fill_value=(1.0, surv_func.values[-1])
            )
            survival_probs[i, :] = interp(times)
            
        return survival_probs.T # 返回形状 [n_samples, n_times]
