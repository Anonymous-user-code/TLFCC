'''
Preset realistic calibration curves for simulated datasets
Reference paper: Combining Priors with Experience: Confidence Calibration Based on Binomial Process Modeling
'''
import numpy as np
import mpmath
from scipy.special import beta as betafn
import matplotlib.pyplot as plt

def logit(s):
    s_clipped = np.clip(s, 1e-10, 1 - 1e-10)
    if isinstance(s,np.ndarray):
        return np.log(s_clipped / (1 - s_clipped))
    else:
        return mpmath.log(s_clipped / (1 - s_clipped))

def sigmoid(s):
    if isinstance(s,np.ndarray):
        return 1/(1+np.exp(-s))
    else:
        return 1/(1+mpmath.exp(-s))

def logflip(s):
    s_clipped = np.clip(s, 1e-10, 1 - 1e-10)
    if isinstance(s,np.ndarray):
        return np.log(1 - s_clipped)
    else:
        return mpmath.log(1 - s_clipped)

def logflip_inverse(s):

    if isinstance(s,np.ndarray):
        return 1-np.exp(s)
    else:
        return 1-mpmath.exp(s)

def log(s):
    s_clipped = np.clip(s, 1e-10, 1 - 1e-10)
    if isinstance(s,np.ndarray):
        return np.log(s_clipped)
    else:
        return mpmath.log(s_clipped)

def log_inverse(s):
    if isinstance(s,np.ndarray):
        return np.exp(s)
    else:
        return mpmath.exp(s)

# D1
class logit_logit1:
    '''
    link functions is logit
    transform functions is logit
    Reference: Section 6 in Mitigating Bias in Calibration Error Estimation
    '''
    def __init__(self) -> None:
        self.beta0 = -0.88
        self.beta1 = 0.49

    def __call__(self,x):
        if isinstance(x,list):
            x = np.array(x)
        p = sigmoid(self.beta0+self.beta1*logit(x))
        return p

# D2
class logflip_logflip:
    '''
    link functions is logflip
    transform functions is logflip
    Reference: Section 6 in Mitigating Bias in Calibration Error Estimation
    '''
    def __init__(self, beta0 = -0.12, beta1 = 0.85) -> None:
        self.beta0 = beta0
        self.beta1 = beta1

    def __call__(self,x):
        if isinstance(x,list):
            x = np.array(x)
        p = logflip_inverse(self.beta0+self.beta1*logflip(x))
        return p

# D3
class log_log:
    '''
    link functions is log
    transform functions is log
    Reference: Section 6 in Mitigating Bias in Calibration Error Estimation
    '''
    def __init__(self, beta0 = -0.03, beta1 = 1.27) -> None:
        self.beta0 = beta0
        self.beta1 = beta1

    def __call__(self,x):
        if isinstance(x,list):
            x = np.array(x)
        p = log_inverse(self.beta0+self.beta1*log(x))
        return p

# D4
class logit_logflip:
    '''
    link functions is logit
    transform functions is logflip
    Reference: Section 6 in Mitigating Bias in Calibration Error Estimation
    '''
    def __init__(self, beta0 = -0.77, beta1 = -0.80) -> None:
        self.beta0 = beta0
        self.beta1 = beta1

    def __call__(self,x):
        if isinstance(x,list):
            x = np.array(x)
        p = sigmoid(self.beta0+self.beta1*logflip(x))
        return p

# D5
class logit_logit2:
    '''
    link functions is logit
    transform functions is logit
    Reference: Section 6 in Mitigating Bias in Calibration Error Estimation
    '''
    def __init__(self) -> None:
        self.beta0 = -0.97
        self.beta1 = 0.34

    def __call__(self,x):
        if isinstance(x,list):
            x = np.array(x)
        p = sigmoid(self.beta0+self.beta1*logit(x))
        return p

class logflip_logflip2:
    '''
    link functions is logflip
    transform functions is logflip
    Reference: Section 6 in Mitigating Bias in Calibration Error Estimation
    '''
    def __init__(self, beta0 = -0.20, beta1 = 0.70) -> None:
        self.beta0 = beta0
        self.beta1 = beta1

    def __call__(self,x):
        if isinstance(x,list):
            x = np.array(x)
        p = logflip_inverse(self.beta0+self.beta1*logflip(x))
        return p
    
class log_log2:
    '''
    link functions is log
    transform functions is log
    Reference: Section 6 in Mitigating Bias in Calibration Error Estimation
    '''
    def __init__(self, beta0 = -0.05, beta1 = 2.52) -> None:
        self.beta0 = beta0
        self.beta1 = beta1

    def __call__(self,x):
        if isinstance(x,list):
            x = np.array(x)
        p = log_inverse(self.beta0+self.beta1*log(x))
        return p
    
class log_log3:
    '''
    link functions is log
    transform functions is log
    Reference: Section 6 in Mitigating Bias in Calibration Error Estimation
    '''
    def __init__(self, beta0 = -0.02, beta1 = 2.12) -> None:
        self.beta0 = beta0
        self.beta1 = beta1

    def __call__(self,x):
        if isinstance(x,list):
            x = np.array(x)
        p = log_inverse(self.beta0+self.beta1*log(x))
        return p
    
class logflip_logflip3:
    '''
    link functions is logflip
    transform functions is logflip
    Reference: Section 6 in Mitigating Bias in Calibration Error Estimation
    '''
    def __init__(self, beta0 = -0.2, beta1 = 0.75) -> None:
        self.beta0 = beta0
        self.beta1 = beta1

    def __call__(self,x):
        if isinstance(x,list):
            x = np.array(x)
        p = logflip_inverse(self.beta0+self.beta1*logflip(x))
        return p
    
class logit_logit3:
    '''
    link functions is logit
    transform functions is logit
    Reference: Section 6 in Mitigating Bias in Calibration Error Estimation
    '''
    def __init__(self) -> None:
        self.beta0 = -0.90
        self.beta1 = 0.56

    def __call__(self,x):
        if isinstance(x,list):
            x = np.array(x)
        p = sigmoid(self.beta0+self.beta1*logit(x))
        return p

class logit_logflip2:
    '''
    link functions is logit
    transform functions is logflip
    Reference: Section 6 in Mitigating Bias in Calibration Error Estimation
    '''
    def __init__(self, beta0 = -0.55, beta1 = -0.90) -> None:
        self.beta0 = beta0
        self.beta1 = beta1

    def __call__(self,x):
        if isinstance(x,list):
            x = np.array(x)
        p = sigmoid(self.beta0+self.beta1*logflip(x))
        return p

class beta_density:
    def __init__(self,alpha = 1.,beta = 1.0) -> None:
        self.alpha = alpha
        self.beta = beta

    def __call__(self,s):
        if isinstance(s,list):
            s = np.array(s)
        s_clipped = np.clip(s, 1e-10, 1 - 1e-10)
        out = (s_clipped**(self.alpha-1))*((1-s_clipped)**(self.beta-1))/betafn(self.alpha,self.beta)
        return out
    
if __name__=="__main__":
    xs = np.linspace(0, 1, 1000)
    ys = logit_logflip2()(xs)
    plt.plot(xs,ys)
    plt.show()