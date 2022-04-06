from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import seaborn
import StudentTMulti as st
import Detector as dt
import hazards as hz
import generate_data as gd
from functools import partial
import pandas as pd
import matplotlib.cm as cm 
from sklearn.neighbors import KernelDensity
from tensorflow import set_random_seed
set_random_seed(2)
import time

class Bayesian_CP():
    
    """
    
    Online Bayesian Changepoint Detection for 
    
    """

    def __init__(self):
        
        start_time = time.time()
        
        # Call Functions
        
        #############################
        ####  Synthetic Data
        #############################

        self.Generate_synthetic_data('5variate')
        #self.Generate_synthetic_data('bivariate')  

        fig=plt.figure(figsize=(12, 8))
        ax1=fig.add_subplot(111)
        ax1.plot(self.data)        
        
        self.dim = self.data.shape[1]
        self.prior = st.StudentTMulti(self.dim)                    
        self.Online_CP_Detection("5variate")

        """
        I/O Data - ICPP Paper - Univariate
        """
        
        for i in range(1,9):
            print i
            data = pd.read_csv("./Data/App"+str(i)+"_Original_new.csv")
            self.data = data.values[:,:]
            self.dim = 1
            self.prior = st.StudentTMulti(self.dim)      
            self.Online_CP_Detection("univariate_Original_App"+str(i))

        
        """
        I/O Data - ICPP Paper - Multivariate
        
        """
        
        ###################
        ### 8-Variate CP
        ####################
        self.Df_Array_app_fillNan = pd.read_csv("./Data/All_8_Benchmarks_time_interpolated.csv")
        self.Df_Array_app_fillNan = self.Df_Array_app_fillNan.apply(lambda x: (x - x.min()) / (x.max() - x.min()))
        self.data = self.Df_Array_app_fillNan.values[:,1:]
        self.dim = self.data.shape[1]
        self.prior = st.StudentTMulti(self.dim)                           
        self.Online_CP_Detection("8variateCP")
                    

        """
        Other Examples go here
        
        """        

       
        print("---Run time  %s seconds ---" % (time.time() - start_time))

    def Online_CP_Detection(self,Appname):
        
        visualize_online = False
        
        detector = dt.Detector()

        if (visualize_online):
            plt.axis([0, len(self.data), np.min(self.data), np.max(self.data)])
            plt.ion()
  
        R_mat = np.zeros((self.data.shape[0],self.data.shape[0]))
        R_mat_cumfreq = np.zeros((self.data.shape[0],self.data.shape[0]))
        R_mat.fill(np.nan)

        for t, x in enumerate(self.data[:,:]):
            #print (t)
            detector.detect(x,partial(hz.constant_hazard,lam=250),self.prior)
            maxes, CP, theta, pred_save_mat = detector.retrieve(self.prior)
            if (visualize_online):
                detector.plot_data_CP_with_Mean(x,maxes,CP,theta)
            R_old = detector.R_old

            try:
                R_mat[t,0:len(R_old)] = R_old
                R_mat_cumfreq[t,0:len(R_old)] = np.cumsum(R_old)
            except:
                R_mat[t,0:len(R_old)] = R_old[0:-1]
                R_mat_cumfreq[t,0:len(R_old)] = np.cumsum(R_old[0:-1])

        R_mat2 = R_mat.copy()
        R_mat = R_mat.T
  

        R_mat_cumfreq = R_mat_cumfreq.T
        R_mat_median = np.nanmedian(R_mat_cumfreq,axis=1)
  
        T = R_mat.shape[1]
        Mrun = np.zeros(T)
        for ii in range(T):            
            try:
                Mrun[ii] = np.where(R_mat_cumfreq[:, ii] >= 0.5)[0][0]
            except:
                pass

        MchangeTime = np.asarray(range(T)) - Mrun + 1

        #########################################################################
        # Find the max value in Mrun sequentially
        # Check if the next value dropped a certain relative value
        # Check if that drop sustains for 10 points   
        CP_CDF = [0]
        for i in xrange(len(Mrun)-5):
            j = i+1
            if ((Mrun[i] - Mrun[j])>5):
                cnt = 0
                for k in xrange(1,20):
                    if (i+k < T) & ((Mrun[i] - Mrun[i+k])> 10):
                        cnt = cnt+1
                    else:
                        break
                if (cnt > 10):
                    CP_CDF.append(i+1)
                    
        fig = plt.figure(figsize=[18, 12])
        ax1 = fig.add_subplot(2, 1, 1)
        ax3 = fig.add_subplot(2, 1, 2)
        sparsity = 1  
        intsty = -np.log(R_mat_cumfreq[0:-1:sparsity, 0:-1:sparsity])
        intsty[np.isnan(intsty)] = 60
        c=ax3.pcolor(np.array(range(0, len(R_mat_cumfreq[:,0]), sparsity)),np.array(range(0, len(R_mat_cumfreq[:,0]), sparsity)), 
                  intsty, 
                  cmap=cm.YlGn_r, vmin=0, vmax=1)
        cb = fig.colorbar(c, ax=ax3,orientation="horizontal", pad=0.2, aspect=70)
        cb.ax.set_xticklabels(cb.ax.get_xticklabels(), fontsize=16)
        cb.ax.set_yticklabels(cb.ax.get_yticklabels(), fontsize=16)
        ax3.plot(Mrun,color='r',marker='.',label='median')
        
        for k in range(self.dim):
            ax1.plot(self.data[:,k])
            
        for num in xrange(len(CP_CDF)):
            ax1.axvline(CP_CDF[num],color='m')
        ax1.set_ylabel("Response", fontsize=16)
        ax3.set_xlabel("Time", fontsize=16)
        ax3.set_ylabel("Run Length", fontsize=16)
        ax1.tick_params(axis='both', which='major', labelsize=14)
        ax3.tick_params(axis='both', which='major', labelsize=14)
        ax3.set_ylim([0,T])
        ax1.set_xlim([0,T])
        fig.savefig("./Plots/"+Appname+"_CDF.pdf", bbox_inches='tight')
        plt.close(fig)
        

        fig = plt.figure(figsize=[18, 12])
        ax = fig.add_subplot(2, 1, 1)
        ax.plot(self.data)
        for num in xrange(len(CP)):
            ax.axvline(CP[num],color='m')    
        ax.set_xlim([0,T])
        ax.set_ylabel("Response", fontsize=16)
        ax.set_xlabel("Time", fontsize=16)
        ax.tick_params(axis='both', which='major', labelsize=14)
        ax = fig.add_subplot(2, 1, 2)
        sparsity = 1  # only plot every fifth data for faster display
        intsty = -np.log(R_mat[0:-1:sparsity, 0:-1:sparsity])
        intsty[intsty == np.inf] = 30
        c = ax.pcolor(np.array(range(0, len(R_mat[:,0]), sparsity)), 
                  np.array(range(0, len(R_mat[:,0]), sparsity)), 
                  intsty, 
                  cmap=cm.gist_gray, vmin=0, vmax=30)
        ax.set_ylabel("Run Length", fontsize=16)
        ax.set_xlabel("Time", fontsize=16)
        ax.tick_params(axis='both', which='major', labelsize=14)
        
        for num in xrange(len(CP)):
            ax.axvline(CP[num],color='m')                       
        fig.savefig("./Plots/"+Appname+"_logPDF.pdf", bbox_inches='tight')
        plt.close(fig)                 

        print "Changepoints locations with PDF:"
        print CP
        print "Changepoints locations with CDF:"
        print CP_CDF

        
    def Generate_synthetic_data(self,flag):        
        if (flag == 'univariate'):
          mean_mat = [10,10,10] #[2,4,6]
          std_mat = [1,3,6]
          data = []
          for i in range(len(mean_mat)):
              data = np.hstack((data,np.random.normal(loc=mean_mat[i],scale=std_mat[i],size =200)))
          data = data.reshape(-1,1)
          
          self.data = data
          
        elif (flag == 'bivariate'):
          mean_mat = [[2,4],[2,4],[2,4]] #[[2,2],[6,2],[6,10]]
          Cov_mat = [[[1,0.9],[0.9,1]],[[3,2.7],[2.7,3]],[[6,0],[0,6]]]
          
          print (mean_mat[0], Cov_mat[0])
          
          data = np.random.multivariate_normal(mean_mat[0],Cov_mat[0],200).T
          
          for i in range(1,len(mean_mat)):
              d2 = np.random.multivariate_normal(mean_mat[i],Cov_mat[i],200).T
              data = np.hstack((data,d2))
          data = data.T 
          self.data = data
          #fig, (ax, ax1, ax2) = plt.subplots(1, 3, figsize=(18, 6))
          #ax.scatter(data[:,0],data[:,1])
          #ax1.hist(data[:,0],bins=20)
          #ax2.hist(data[:,1],bins=20)
          
        else:
          dim = 5
          partition, data = gd.generate_multinormal_time_series(4, dim, 100, 300)          
          changes = np.cumsum(partition)

          fig, ax = plt.subplots(figsize=[16,12])
          for p in changes:
              ax.plot([p,p],[np.min(data),np.max(data)],'r')
          for d in range(dim):
              ax.plot(data[:,d])
          plt.show()
    
          #data = data.T
          fig, (ax, ax1, ax2) = plt.subplots(1, 3, figsize=(18, 6))
          ax.scatter(data[:,0],data[:,1])
          ax1.hist(data[:,0],bins=20)
          ax2.hist(data[:,1],bins=20) 
          
          self.data = data       

        
if __name__ == '__main__':

    Test_CP = Bayesian_CP()
          
