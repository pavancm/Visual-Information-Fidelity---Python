import numpy as np
from numpy.linalg import inv
import pyPyrTools as ppt
from pyPyrTools.corrDn import corrDn
import math

def vifvec(imref_batch,imdist_batch):
    M = 3
    subbands = [4, 7, 10, 13, 16, 19, 22, 25]
    sigma_nsq = 0.4
    
    batch_num =1
    if imref_batch.ndim >= 3: 
        batch_num = imref_batch.shape[0]
    
    vif = np.zeros([batch_num,])
    
    for a in range(batch_num):
        if batch_num > 1:
            imref = imref_batch[a,:,:]
            imdist = imdist_batch[a,:,:]
        else:
            imref = imref_batch
            imdist = imdist_batch
        
        #Wavelet Decomposition
        pyr = ppt.Spyr(imref, 4, 'sp5Filters', 'reflect1')
        org = pyr.pyr[::-1]     #reverse list
        
        pyr = ppt.Spyr(imdist, 4, 'sp5Filters', 'reflect1')
        dist = pyr.pyr[::-1]
        
        #Calculate parameters of the distortion channel
        g_all, vv_all = vif_sub_est_M(org, dist, subbands, M)
        
        #calculate the parameters of reference
        ssarr, larr, cuarr = refparams_vecgsm(org, subbands, M)
        
        num = np.zeros([1,len(subbands)])
        den = np.zeros([1,len(subbands)])
        
        for i in range(len(subbands)):
            sub = subbands[i]
            g = g_all[i]
            vv = vv_all[i]
            ss = ssarr[i]
            lam = larr[i]
            #cu = cuarr[i]
            
            #neigvals = len(lam)
            lev = math.ceil((sub - 1)/6)
            winsize = 2**lev + 1
            offset = (winsize - 1)/2
            offset = math.ceil(offset/M)
            
            g = g[offset:g.shape[0]-offset,offset:g.shape[1]-offset]
            vv = vv[offset:vv.shape[0]-offset,offset:vv.shape[1]-offset]
            ss = ss[offset:ss.shape[0]-offset,offset:ss.shape[1]-offset]
            
            temp1,temp2 = 0,0
            rt = []
            for j in range(len(lam)):
                temp1 += np.sum(np.log2(1 + np.divide(np.multiply(np.multiply(g,g),ss) * lam[j], vv + sigma_nsq))) #distorted image information
                temp2 += np.sum(np.log2(1 + np.divide(ss * lam[j], sigma_nsq))) #reference image information
                rt.append(np.sum(np.log(1 + np.divide(ss * lam[j], sigma_nsq))))
            
            num[0,i] = temp1
            den[0,i] = temp2
        
        vif[a] = np.sum(num)/np.sum(den)
    print(vif)
    return vif
        

def vif_sub_est_M(org, dist, subbands, M):
    tol = 1e-15         #tolerance for zero variance
    g_all = []
    vv_all = []
    
    for i in range(len(subbands)):
        sub = subbands[i]
        y = org[sub-1]
        yn = dist[sub-1]
        
        #size of window used in distortion channel estimation
        lev = math.ceil((sub - 1)/6)
        winsize = 2**lev + 1
        win = np.ones([winsize, winsize])
        
        #force subband to be a multiple of M
        newsize = [math.floor(y.shape[0]/M) * M, math.floor(y.shape[1]/M) * M]
        y = y[:newsize[0],:newsize[1]]
        yn = yn[:newsize[0],:newsize[1]]
        
        #correlation with downsampling
        winstep = (M, M)
        winstart = (math.floor(M/2) ,math.floor(M/2))
        winstop = (y.shape[0] - math.ceil(M/2) + 1, y.shape[1] - math.ceil(M/2) + 1)
        
        #mean
        mean_x = corrDn(y, win/np.sum(win), 'reflect1', winstep, winstart, winstop)
        mean_y = corrDn(yn, win/np.sum(win), 'reflect1', winstep, winstart, winstop)
        
        #covariance
        cov_xy = corrDn(np.multiply(y, yn), win, 'reflect1', winstep, winstart, winstop) - \
        np.sum(win) * np.multiply(mean_x,mean_y)
        
        #variance
        ss_x = corrDn(np.multiply(y,y), win, 'reflect1', winstep, winstart, winstop) - np.sum(win) * np.multiply(mean_x,mean_x)
        ss_y = corrDn(np.multiply(yn,yn), win, 'reflect1', winstep, winstart, winstop) - np.sum(win) * np.multiply(mean_y, mean_y)
        
        ss_x[np.where(ss_x <0)] = 0
        ss_y[np.where(ss_y <0)] = 0
        
        #Regression
        g = np.divide(cov_xy,(ss_x + tol))
        
        vv = (ss_y - np.multiply(g, cov_xy))/(np.sum(win))
        
        g[np.where(ss_x < tol)] = 0
        vv[np.where(ss_x < tol)] = ss_y[np.where(ss_x < tol)]
        ss_x[np.where(ss_x < tol)] = 0
        
        g[np.where(ss_y < tol)] = 0
        vv[np.where(ss_y < tol)] = 0
        
        g[np.where(g < 0)] = 0
        vv[np.where(g < 0)] = ss_y[np.where(g < 0)]
        
        vv[np.where(vv <= tol)] = tol
        
        g_all.append(g)
        vv_all.append(vv)
    
    return g_all, vv_all

def refparams_vecgsm(org, subbands, M):
    # This function caluclates the parameters of the reference image
    #l_arr = np.zeros([subbands[-1],M**2])
    l_arr, ssarr, cu_arr = [],[],[]
    for i in range(len(subbands)):
        sub = subbands[i]
        y = org[sub-1]
        
        sizey = (math.floor(y.shape[0]/M)*M, math.floor(y.shape[1]/M)*M)
        y = y[:sizey[0],:sizey[1]]
        
        #Collect MxM blocks, rearrange into M^2 dimensional vector
        temp = []
        for j in range(M):
            for k in range(M):
                temp.append(y[k:y.shape[0]-M+k+1,j:y.shape[1]-M+j+1].T.reshape(-1))
        
        temp = np.asarray(temp)
        mcu = np.mean(temp, axis=1).reshape(temp.shape[0],1)
        mean_sub = temp - np.repeat(mcu,temp.shape[1],axis=1)
        cu = mean_sub @ mean_sub.T / temp.shape[1]
        #Calculate S field, non-overlapping blocks
        temp = []
        for j in range(M):
            for k in range(M):
                temp.append(y[k::M,j::M].T.reshape(-1))
        
        temp = np.asarray(temp)
        ss = inv(cu) @ temp
        ss = np.sum(np.multiply(ss,temp),axis=0)/(M**2)
        ss = ss.reshape(int(sizey[1]/M), int(sizey[0]/M)).T
        
        d, _ = np.linalg.eig(cu)
        l_arr.append(d)
        ssarr.append(ss)
        cu_arr.append(cu)
    
    return ssarr, l_arr, cu_arr