import numpy as np
from scipy.optimize import curve_fit
import math
import pandas as pd
from statsmodels.formula.api import glm
import statsmodels.api as sm

class Histom:
    def __init__(self,Ds,Dt,K,bin_num=10):
        '''
        Ds: Source domain data
        Dt: Target domain data
        K: The number of class
        '''
        self.Ds = Ds
        self.Dt = Dt
        self.K = K
        
        #get hat_S in Dt
        self.hat_S_in_Dt = []
        for sample in self.Dt:
            self.hat_S_in_Dt.append(sample[0])

        #get hat_D_t_k
        self.hat_D_t_ks = []
        for k in range(K):
            hat_D_t_k = []
            for sample in self.Dt:
                if sample[2]==k:
                    hat_D_t_k.append(sample[0])
            self.hat_D_t_ks.append(hat_D_t_k)

        #get D_s_k and no_D_s_k
        self.D_s_ks = []
        self.no_D_s_ks = []
        for k in range(K):
            D_s_k = []
            no_D_s_k = []
            for sample in self.Ds:
                if sample[3]==k and sample[2] == k:
                    D_s_k.append(sample[0])
                elif sample[3]!=k and sample[2] == k:
                    no_D_s_k.append(sample[0])
            self.D_s_ks.append(D_s_k)
            self.no_D_s_ks.append(no_D_s_k)

        bins = np.linspace(0, 1, bin_num+1)
        self.bins = bins

        #Estimating
        self.g_2, self.g_1_ks, self.g_3_ks, self.g_4_ks = self.Estimating()

    def Estimating(self):

        counts, a = np.histogram(self.hat_S_in_Dt, bins=self.bins)
        g_2 = counts/len(self.hat_S_in_Dt)
        x = [(a[i]+a[i+1])/2 for i in range(len(a)-1)]

        g_1_ks = []
        g_3_ks = []
        g_4_ks = []

        for k in range(self.K):
            
            counts, _ = np.histogram(self.hat_D_t_ks[k], bins=self.bins)
            g_3_k = counts/len(self.hat_D_t_ks[k])

            counts, _ = np.histogram(self.D_s_ks[k], bins=self.bins)
            g_1_k = counts/len(self.D_s_ks[k])

            if self.no_D_s_ks[k] != []: 
                counts, _ = np.histogram(self.no_D_s_ks[k], bins=self.bins)
                g_4_k = counts/len(self.no_D_s_ks[k])
            else:
                g_4_k = np.array([0. for i in range(len(self.bins)-1)])

            # plt.plot(x,g_1_k, label="g1")
            # plt.plot(x,g_2, label="g2")
            # plt.plot(x,g_3_k, label = "g3")
            # plt.plot(x,g_4_k, label = "g4")
            # plt.ylim([0,1])
            # plt.legend()
            # plt.show()

            g_1_ks.append(g_1_k)
            g_3_ks.append(g_3_k)
            g_4_ks.append(g_4_k)
        
        return g_2, g_1_ks, g_3_ks, g_4_ks
    
    def Calibrating(self):
        D_cali = []
        for i in range(len(self.g_2)):
            hat_S = 0.5*(self.bins[i+1]+self.bins[i])
            cali_result = 0.
            for k in range(self.K):
                hat_N_t_k = len(self.hat_D_t_ks[k])/len(self.hat_S_in_Dt)
                if self.g_2[i] != 0 and abs(self.g_1_ks[k][i]-self.g_4_ks[k][i]) >= 0.0005:
                    cali_result = cali_result + hat_N_t_k * (self.g_1_ks[k][i]/self.g_2[i]) * (self.g_3_ks[k][i]-self.g_4_ks[k][i])/(self.g_1_ks[k][i]-self.g_4_ks[k][i])
                    if k == self.K -1:
                        D_cali.append([hat_S,cali_result])
                else:
                    break
        return D_cali
    
    def Fitting(self, xs,ys):
        bounds = ([0, 0, -100], [100, 100, 100])
        params, covariance = curve_fit(self.prior_fun, xs, ys, bounds=bounds)
        a_fit, b_fit, c_fit = params
        return a_fit, b_fit, c_fit

    def prior_fun(self,x, a, b, c):
        return 1/(1+(x**(-a))*((1-x)**b)*math.exp(c))
    
    def glm_fit(self,xs,ys):
        
        data = pd.DataFrame({'x': xs, 'y': ys})


        formulas = [
            "y ~ x + I(x**2)", 
            "y ~ x + I(x**3)", 
            "y ~ I(x**2) + I(x**3)",  
            "y ~ x + I(x**2) + I(x**3)",  
            "y ~ x + I(x**2) + I(x**3)+I(x**4)",
            "y ~ x +I(x**4)",
            "y ~ I(x**2) + I(x**4)",
            "y ~ I(x**3)+I(x**4)",
        ]

        
        # 拟合模型并计算AIC
        models = []
        for formula in formulas:
            model = glm(formula=formula, data=data, family=sm.families.Gaussian()).fit()
            models.append((model, model.aic))

        # 选择AIC最小的模型
        scores = [model[1] for model in models]
        best_index = np.argmin(scores)
        best_model = models[best_index]

        print("Best_index:",best_index)
        print("Best_AIC:",scores[best_index])

        transform_predict_fun = None

        return best_model[0].predict, transform_predict_fun

