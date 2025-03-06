import random
import numpy as np
import True_calibration_curves
import matplotlib.pyplot as plt

class Simulated_data:
    def __init__(self,Rs,Rt,alpha1,beta1,alpha2,beta2,g1,g2):
        '''
        Ns: Sample size in source domain
        Nt: Sample size in target domain
        Rs: Class probability of the first class in source domain: $P(Y=1)$
        Rt: Class probability of the first class in Target domain: $Q(Y=1)$
        alpha1,beta1: Beta distribution parameters of the confidence distribution of the first class: $Be(alpha1,beta1)$
        alpha2,beta2: Beta distribution parameters of the confidence distribution of the second class: $Be(alpha2,beta2)$
        g1: The preset true calibration curve of the first class
        g2: The preset true calibration curve of the second class
        '''
        self.Ns = 100000
        self.Nt = 100000
        self.Rs = Rs
        self.Rt = Rt
        self.alpha1,self.beta1 = alpha1, beta1
        self.alpha2,self.beta2 = alpha2, beta2
        self.g1 = g1
        self.g2 = g2
        
    def get_dataset(self,domain = "source"):   
        '''
        domain: source or target
        '''
        if domain == "source":
            Y_rate = self.Rs
        elif domain == "target":
            Y_rate = self.Rt

        Ds = []
        i = 1
        while i <= self.Ns: 
            if random.random() <= Y_rate:
                hat_S = np.random.beta(self.alpha1,self.beta1,size=1)
                hat_S = np.clip(hat_S,1e-4,1-1e-4)
                H = np.random.binomial(1, self.g1(hat_S))
                if H == 1:
                    hat_Y = 0
                else:
                    hat_Y = 1
                Ds.append([hat_S[0],H,hat_Y,0])
            else:
                hat_S = np.random.beta(self.alpha2,self.beta2,size=1)
                H = np.random.binomial(1, self.g2(hat_S))
                if H == 1:
                    hat_Y = 1
                else:
                    hat_Y = 0
                Ds.append([hat_S[0],H,hat_Y,1])
            i = i + 1
        Ds = sorted(Ds, key=lambda item: item[0])
        return Ds

    def get_true_curve(self,hat_S, domain = "source"):
        '''
        domain: source or target
        '''
        if domain == "source":
            Y_rate = self.Rs
        elif domain == "target":
            Y_rate = self.Rt

        density1 = True_calibration_curves.beta_density(self.alpha1,self.beta1)
        density2 = True_calibration_curves.beta_density(self.alpha2,self.beta2)
        cali_value = self.g1(hat_S)*Y_rate*density1(hat_S)/(density1(hat_S)*Y_rate+density2(hat_S)*(1-Y_rate)) + self.g2(hat_S)*(1-Y_rate)*density2(hat_S)/(density1(hat_S)*Y_rate+density2(hat_S)*(1-Y_rate))
        return cali_value
    
    def plot_effect_of_label_shift(self):
        # plot true curve
        x = np.linspace(0, 1, 1000)
        # plot the true curve of source domain
        y = self.get_true_curve(x)
        plt.plot(x,y,label = f"Source domain ($P(Y=0)$={self.Rs})")
        # plot the true curve of target domain
        y = self.get_true_curve(x,domain="target")
        plt.plot(x,y,label = f"Target domain ($Q(Y=0)$={self.Rt})")
        fontsize = 20
        plt.xlabel(f'Confidence score',fontname="Times New Roman",fontsize=fontsize)
        plt.ylabel(f'Calibration value',fontname="Times New Roman",fontsize=fontsize)
        plt.legend(prop={"family": "Times New Roman","size":fontsize},loc="upper left",ncol = 1,frameon=False,columnspacing=0.1)
        plt.tick_params(axis='both', labelsize=fontsize)
        plt.rcParams['font.family'] = 'Times New Roman'
        plt.xlim([0,1])
        plt.ylim([0,1])
        plt.show()
    
    def valid_true_curve(self,domain = "source"):
        '''
        domain: source or target
        '''

        Ds = self.get_dataset(domain)

        # plot histogram fitting in Ds
        xs = []
        ys = []
        for i in range(len(Ds)):
            xs.append(Ds[i][0])
            ys.append(Ds[i][1])
        xs = np.array(xs)
        ys = np.array(ys)
        bins = np.linspace(0, 1, 21)  
        bin_centers = 0.5 * (bins[:-1] + bins[1:])  
        y_prob = []
        for i in range(len(bins) - 1):
            mask = (xs >= bins[i]) & (xs < bins[i + 1])
            if mask.sum() > 0:
                y_prob.append(ys[mask].mean())
            else:
                y_prob.append(np.nan)
        plt.plot(bin_centers,y_prob)

        # plot true curve
        x = np.linspace(0, 1, 1000)
        y = self.get_true_curve(x)
        plt.plot(x,y)
        plt.show()
