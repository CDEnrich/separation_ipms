import matplotlib
matplotlib.use('Agg')

import glob
import numpy as np
import os
import pickle
import torch
import matplotlib.pyplot as plt

plt.style.use('ggplot')

class empty_class:
    pass

def values_f1_f2_sep(args):
    d_f1_avg = []
    d_f2_avg = []
    d_f1_t_avg = []
    d_f1_nu_avg = []
    d_f2_nu_avg = []
    d_f1_max = []
    d_f2_max = []
    d_f1_t_max = []
    d_f1_nu_max = []
    d_f2_nu_max = []
    d_f1_min = []
    d_f2_min = []
    d_f1_t_min = []
    d_f1_nu_min = []
    d_f2_nu_min = []
    ratio = []
    ratio_max = []
    ratio_min = []
    ratio_nu = []
    ratio_nu_max = []
    ratio_nu_min = []
    sqrt_N_kd_list = []
    for dimension in args.d_vec:
        args.d = dimension
        resdir = os.path.join('res', args.name)
        name = os.path.join(resdir,f'{args.name}_{args.d}_{args.k}_{args.n_samples}_{args.n_feature_samples}_*_{args.alpha}_{args.gamma}_{args.a}_{args.b}.pkl')
        fnames = glob.glob(name)
        d_f1_list = []
        d_f2_list = []
        d_f1_t_list = []
        d_f1_nu_list = []
        d_f2_nu_list = []
        sqrt_N_kd = 0
        assert len(fnames) > 0, 'no files! ({})'.format(name)
        for fname in fnames:
            res = pickle.load(open(fname, 'rb'))
            d_f1_list.append(res['d_f1'])
            d_f2_list.append(res['d_f2'])
            d_f1_t_list.append(res['d_f1_t'])
            d_f1_nu_list.append(res['d_f1_nu'])
            d_f2_nu_list.append(res['d_f2_nu'])
            sqrt_N_kd = res['sqrt(N_kd)']
        d_f1_list = torch.tensor(d_f1_list)
        d_f2_list = torch.tensor(d_f2_list)
        d_f1_t_list = torch.tensor(d_f1_t_list)
        d_f1_nu_list = torch.tensor(d_f1_nu_list)
        d_f2_nu_list = torch.tensor(d_f2_nu_list)
        d_f1_avg.append(float(torch.mean(d_f1_list)))
        d_f2_avg.append(float(torch.mean(d_f2_list)))
        d_f1_t_avg.append(float(torch.mean(d_f1_t_list)))
        d_f1_nu_avg.append(float(torch.mean(d_f1_nu_list)))
        d_f2_nu_avg.append(float(torch.mean(d_f2_nu_list)))
        d_f1_max.append(float(torch.max(d_f1_list)))
        d_f2_max.append(float(torch.max(d_f2_list)))
        d_f1_t_max.append(float(torch.max(d_f1_t_list)))
        d_f1_nu_max.append(float(torch.max(d_f1_nu_list)))
        d_f2_nu_max.append(float(torch.max(d_f2_nu_list)))
        d_f1_min.append(float(torch.min(d_f1_list)))
        d_f2_min.append(float(torch.min(d_f2_list)))
        d_f1_t_min.append(float(torch.min(d_f1_t_list)))
        d_f1_nu_min.append(float(torch.min(d_f1_nu_list)))
        d_f2_nu_min.append(float(torch.min(d_f2_nu_list)))
        ratio.append(float(torch.mean(d_f1_list)/torch.mean(d_f2_list)))
        ratio_max.append(float(torch.max(d_f1_list)/torch.min(d_f2_list)))
        ratio_min.append(float(torch.min(d_f1_list)/torch.max(d_f2_list)))
        ratio_nu.append(float(torch.mean(d_f1_nu_list)/torch.mean(d_f2_nu_list)))
        ratio_nu_max.append(float(torch.max(d_f1_nu_list)/torch.min(d_f2_nu_list)))
        ratio_nu_min.append(float(torch.min(d_f1_nu_list)/torch.max(d_f2_nu_list)))
        sqrt_N_kd_list.append(sqrt_N_kd)
    dict_lists = {
            'd_f1_avg': d_f1_avg,
            'd_f2_avg': d_f2_avg,
            'd_f1_t_avg': d_f1_t_avg,
            'd_f1_nu_avg': d_f1_nu_avg,
            'd_f2_nu_avg': d_f2_nu_avg,
            'd_f1_max': d_f1_max,
            'd_f2_max': d_f2_max,
            'd_f1_t_max': d_f1_t_max,
            'd_f1_nu_max': d_f1_nu_max,
            'd_f2_nu_max': d_f2_nu_max,
            'd_f1_min': d_f1_min,
            'd_f2_min': d_f2_min,
            'd_f1_t_min': d_f1_t_min,
            'd_f1_nu_min': d_f1_nu_min,
            'd_f2_nu_min': d_f2_nu_min,
            'ratio': ratio,
            'ratio_max': ratio_max,
            'ratio_min': ratio_min,
            'ratio_nu': ratio_nu,
            'ratio_nu_max': ratio_nu_max,
            'ratio_nu_min': ratio_nu_min,
            'sqrt_N_kd_list': sqrt_N_kd_list,
    }
    return dict_lists

def fig_1_f1_f2_sep(args, results):
    d_axis = np.array(args.d_vec)
    d_f1_avg = np.array(results['d_f1_avg'])
    d_f2_avg = np.array(results['d_f2_avg'])
    d_f1_t_avg = np.array(results['d_f1_t_avg'])
    d_f1_max = np.array(results['d_f1_max'])
    d_f2_max = np.array(results['d_f2_max'])
    d_f1_t_max = np.array(results['d_f1_t_max'])
    d_f1_min = np.array(results['d_f1_min'])
    d_f2_min = np.array(results['d_f2_min'])
    d_f1_t_min = np.array(results['d_f1_t_min'])
    
    plt.semilogy(d_axis, d_f1_avg, label='$F_1$ IPM')
    plt.semilogy(d_axis, d_f1_t_avg, label='$F_1$ IPM (theory)')
    plt.semilogy(d_axis, d_f2_avg, label='$F_2$ IPM')
    plt.fill_between(d_axis, d_f1_min, d_f1_max, alpha=.3)
    plt.fill_between(d_axis, d_f1_t_min, d_f1_t_max, alpha=.3)
    plt.fill_between(d_axis, d_f2_min, d_f2_max, alpha=.3)
    
    plt.ylabel('IPMs')
    plt.xlabel('Dimension d')
    plt.legend()
    
def fig_2_f1_f2_sep(args, results):
    d_axis = np.array(args.d_vec)
    ratio = np.array(results['ratio'])
    ratio_max = np.array(results['ratio_max'])
    ratio_min = np.array(results['ratio_min'])
    sqrt_N_kd_list = np.array(results['sqrt_N_kd_list'])
    
    plt.semilogy(d_axis, ratio, label='Ratio $F_1$ IPM/$F_2$ IPM')
    plt.semilogy(d_axis, sqrt_N_kd_list, label='Ratio (theory)')
    plt.fill_between(d_axis, ratio_min, ratio_max, alpha=.3)
    
    plt.ylabel('Ratios')
    plt.xlabel('Dimension d')
    plt.legend()
    
def values_f1_f2_sd(args):
    sd_f1_avg = []
    sd_f2_avg = []
    sd_f1_t_avg = []
    sd_f1_max = []
    sd_f2_max = []
    sd_f1_t_max = []
    sd_f1_min = []
    sd_f2_min = []
    sd_f1_t_min = []
    sd_ratio = []
    sd_ratio_max = []
    sd_ratio_min = []
    sd_ratio_t_list = []
    for dimension in args.d_vec:
        args.d = dimension
        resdir = os.path.join('res', args.name)
        name = os.path.join(resdir,f'{args.name}_{args.d}_{args.k}_{args.n_samples}_{args.n_feature_samples}_*_{args.alpha}_{args.gamma}_{args.a}_{args.b}.pkl')
        fnames = glob.glob(name)
        sd_f1_list = []
        sd_f2_list = []
        sd_f1_t_list = []
        sd_ratio_t = 0
        assert len(fnames) > 0, 'no files! ({})'.format(name)
        for fname in fnames:
            res = pickle.load(open(fname, 'rb'))
            sd_f1_list.append(res['sd_f1'])
            sd_f2_list.append(res['sd_f2'])
            sd_f1_t_list.append(res['sd_f1_t'])
            sd_ratio_t = res['sd_ratio_t']
        sd_f1_list = torch.tensor(sd_f1_list)
        sd_f2_list = torch.tensor(sd_f2_list)
        sd_f1_t_list = torch.tensor(sd_f1_t_list)
        sd_f1_avg.append(float(torch.mean(sd_f1_list)))
        sd_f2_avg.append(float(torch.mean(sd_f2_list)))
        sd_f1_t_avg.append(float(torch.mean(sd_f1_t_list)))
        sd_f1_max.append(float(torch.max(sd_f1_list)))
        sd_f2_max.append(float(torch.max(sd_f2_list)))
        sd_f1_t_max.append(float(torch.max(sd_f1_t_list)))
        sd_f1_min.append(float(torch.min(sd_f1_list)))
        sd_f2_min.append(float(torch.min(sd_f2_list)))
        sd_f1_t_min.append(float(torch.min(sd_f1_t_list)))
        sd_ratio.append(float(torch.mean(sd_f1_list)/torch.mean(sd_f2_list)))
        sd_ratio_max.append(float(torch.max(sd_f1_list)/torch.min(sd_f2_list)))
        sd_ratio_min.append(float(torch.min(sd_f1_list)/torch.max(sd_f2_list)))
        sd_ratio_t_list.append(sd_ratio_t)
    dict_lists = {
            'sd_f1_avg': sd_f1_avg,
            'sd_f2_avg': sd_f2_avg,
            'sd_f1_t_avg': sd_f1_t_avg, 
            'sd_f1_max': sd_f1_max,
            'sd_f2_max': sd_f2_max,
            'sd_f1_t_max': sd_f1_t_max,
            'sd_f1_min': sd_f1_min,
            'sd_f2_min': sd_f2_min,
            'sd_f1_t_min': sd_f1_t_min,
            'sd_ratio': sd_ratio,
            'sd_ratio_max': sd_ratio_max,
            'sd_ratio_min': sd_ratio_min,
            'sd_ratio_t_list': sd_ratio_t_list,
    }
    return dict_lists

def fig_1_f1_f2_sd(args, results):
    d_axis = np.array(args.d_vec)
    sd_f1_avg = np.array(results['sd_f1_avg'])
    sd_f2_avg = np.array(results['sd_f2_avg'])
    sd_f1_t_avg = np.array(results['sd_f1_t_avg'])
    sd_f1_min = np.array(results['sd_f1_min'])
    sd_f2_min = np.array(results['sd_f2_min'])
    sd_f1_t_min = np.array(results['sd_f1_t_min'])
    sd_f1_max = np.array(results['sd_f1_max'])
    sd_f2_max = np.array(results['sd_f2_max'])
    sd_f1_t_max = np.array(results['sd_f1_t_max'])
    
    plt.semilogy(d_axis, sd_f1_avg, label='$F_1$ SD')
    plt.semilogy(d_axis, sd_f1_t_avg, label='$F_1$ SD (theory)')
    plt.semilogy(d_axis, sd_f2_avg, label='$F_2$ SD')
    plt.fill_between(d_axis, sd_f1_min, sd_f1_max, alpha=.3)
    plt.fill_between(d_axis, sd_f1_t_min, sd_f1_t_max, alpha=.3)
    plt.fill_between(d_axis, sd_f2_min, sd_f2_max, alpha=.3)
    
    plt.ylabel('SDs')
    plt.xlabel('Dimension d')
    plt.legend()
    
def fig_2_f1_f2_sd(args, results):
    d_axis = np.array(args.d_vec)
    sd_ratio = np.array(results['sd_ratio'])
    sd_ratio_max = np.array(results['sd_ratio_max'])
    sd_ratio_min = np.array(results['sd_ratio_min'])
    sd_ratio_t_list = np.array(results['sd_ratio_t_list'])
    
    plt.semilogy(d_axis, sd_ratio, label='Ratio $F_1$ SD/$F_2$ SD')
    plt.semilogy(d_axis, sd_ratio_t_list, label='Ratio lower bound (theory)')
    plt.fill_between(d_axis, sd_ratio_min, sd_ratio_max, alpha=.3)
    
    plt.ylabel('Ratios')
    plt.xlabel('Dimension d')
    plt.legend()
    
def values_f1_f2_sliced_w(args):
    d_f1_avg = []
    d_f2_avg = []
    d_tildef2_avg = []
    max_sliced_avg = []
    avg_sliced_avg = []
    d_f1_max = []
    d_f2_max = []
    d_tildef2_max = []
    max_sliced_max = []
    avg_sliced_max = []
    d_f1_min = []
    d_f2_min = []
    d_tildef2_min = []
    max_sliced_min = []
    avg_sliced_min = []
    d_f1_nu_avg = []
    d_f2_nu_avg = []
    d_tildef2_nu_avg = []
    max_sliced_nu_avg = []
    avg_sliced_nu_avg = []
    d_f1_nu_max = []
    d_f2_nu_max = []
    d_tildef2_nu_max = []
    max_sliced_nu_max = []
    avg_sliced_nu_max = []
    d_f1_nu_min = []
    d_f2_nu_min = []
    d_tildef2_nu_min = []
    max_sliced_nu_min = []
    avg_sliced_nu_min = []
    for dimension in args.d_vec:
        args.d = dimension
        resdir = os.path.join('res', args.name)
        name = os.path.join(resdir,f'{args.name}_{args.d}_{args.n_samples}_{args.n_feature_samples}_*_{args.alpha}_{args.large_var}_{args.small_var}.pkl')
        fnames = glob.glob(name)
        d_f1_list = []
        d_f2_list = []
        d_tildef2_list = []
        max_sliced_list = []
        avg_sliced_list = []
        d_f1_nu_list = []
        d_f2_nu_list = []
        d_tildef2_nu_list = []
        max_sliced_nu_list = []
        avg_sliced_nu_list = []
        assert len(fnames) > 0, 'no files! ({})'.format(name)
        for fname in fnames:
            res = pickle.load(open(fname, 'rb'))
            d_f1_list.append(res['d_f1'])
            d_f2_list.append(res['d_f2'])
            d_tildef2_list.append(res['d_tildef2'])
            max_sliced_list.append(res['max_sliced'])
            avg_sliced_list.append(res['avg_sliced'])
            d_f1_nu_list.append(res['d_f1_nu'])
            d_f2_nu_list.append(res['d_f2_nu'])
            d_tildef2_nu_list.append(res['d_tildef2_nu'])
            max_sliced_nu_list.append(res['max_sliced_nu'])
            avg_sliced_nu_list.append(res['avg_sliced_nu'])
        d_f1_list = torch.tensor(d_f1_list)
        d_f2_list = torch.tensor(d_f2_list)
        d_tildef2_list = torch.tensor(d_tildef2_list)
        max_sliced_list = torch.tensor(max_sliced_list)
        avg_sliced_list = torch.tensor(avg_sliced_list)
        d_f1_nu_list = torch.tensor(d_f1_nu_list)
        d_f2_nu_list = torch.tensor(d_f2_nu_list)
        d_tildef2_nu_list = torch.tensor(d_tildef2_nu_list)
        max_sliced_nu_list = torch.tensor(max_sliced_nu_list)
        avg_sliced_nu_list = torch.tensor(avg_sliced_nu_list)
        
        d_f1_avg.append(float(torch.mean(d_f1_list)))
        d_f2_avg.append(float(torch.mean(d_f2_list)))
        d_tildef2_avg.append(float(torch.mean(d_tildef2_list)))
        max_sliced_avg.append(float(torch.mean(max_sliced_list)))
        avg_sliced_avg.append(float(torch.mean(avg_sliced_list)))
        d_f1_max.append(float(torch.max(d_f1_list)))
        d_f2_max.append(float(torch.max(d_f2_list)))
        d_tildef2_max.append(float(torch.max(d_tildef2_list)))
        max_sliced_max.append(float(torch.max(max_sliced_list)))
        avg_sliced_max.append(float(torch.max(avg_sliced_list)))
        d_f1_min.append(float(torch.min(d_f1_list)))
        d_f2_min.append(float(torch.min(d_f2_list)))
        d_tildef2_min.append(float(torch.min(d_tildef2_list)))
        max_sliced_min.append(float(torch.min(max_sliced_list)))
        avg_sliced_min.append(float(torch.min(avg_sliced_list)))
        d_f1_nu_avg.append(float(torch.mean(d_f1_nu_list)))
        d_f2_nu_avg.append(float(torch.mean(d_f2_nu_list)))
        d_tildef2_nu_avg.append(float(torch.mean(d_tildef2_nu_list)))
        max_sliced_nu_avg.append(float(torch.mean(max_sliced_nu_list)))
        avg_sliced_nu_avg.append(float(torch.mean(avg_sliced_nu_list)))
        d_f1_nu_max.append(float(torch.max(d_f1_nu_list)))
        d_f2_nu_max.append(float(torch.max(d_f2_nu_list)))
        d_tildef2_nu_max.append(float(torch.max(d_tildef2_nu_list)))
        max_sliced_nu_max.append(float(torch.max(max_sliced_nu_list)))
        avg_sliced_nu_max.append(float(torch.max(avg_sliced_nu_list)))
        d_f1_nu_min.append(float(torch.min(d_f1_nu_list)))
        d_f2_nu_min.append(float(torch.min(d_f2_nu_list)))
        d_tildef2_nu_min.append(float(torch.min(d_tildef2_nu_list)))
        max_sliced_nu_min.append(float(torch.min(max_sliced_nu_list)))
        avg_sliced_nu_min.append(float(torch.min(avg_sliced_nu_list)))
    dict_lists = {
            'd_f1_avg': d_f1_avg,
            'd_f2_avg': d_f2_avg,
            'd_tildef2_avg': d_tildef2_avg,
            'max_sliced_avg': max_sliced_avg,
            'avg_sliced_avg': avg_sliced_avg,
            'd_f1_max': d_f1_max,
            'd_f2_max': d_f2_max,
            'd_tildef2_max': d_tildef2_max,
            'max_sliced_max': max_sliced_max,
            'avg_sliced_max': avg_sliced_max,
            'd_f1_min': d_f1_min,
            'd_f2_min': d_f2_min,
            'd_tildef2_min': d_tildef2_min,
            'max_sliced_min': max_sliced_min,
            'avg_sliced_min': avg_sliced_min,
            'd_f1_nu_avg': d_f1_nu_avg,
            'd_f2_nu_avg': d_f2_nu_avg,
            'd_tildef2_nu_avg': d_tildef2_nu_avg,
            'max_sliced_nu_avg': max_sliced_nu_avg,
            'avg_sliced_nu_avg': avg_sliced_nu_avg,
            'd_f1_nu_max': d_f1_nu_max,
            'd_f2_nu_max': d_f2_nu_max,
            'd_tildef2_nu_max': d_tildef2_nu_max,
            'max_sliced_nu_max': max_sliced_nu_max,
            'avg_sliced_nu_max': avg_sliced_nu_max,
            'd_f1_nu_min': d_f1_nu_min,
            'd_f2_nu_min': d_f2_nu_min,
            'd_tildef2_nu_min': d_tildef2_nu_min,
            'max_sliced_nu_min': max_sliced_nu_min,
            'avg_sliced_nu_min': avg_sliced_nu_min,
    }
    return dict_lists

def fig_1_f1_f2_sliced_w(args, results):
    d_axis = np.array(args.d_vec)
    d_f1_avg = np.array(results['d_f1_avg'])
    d_f2_avg = np.array(results['d_f2_avg'])
    d_tildef2_avg = np.array(results['d_tildef2_avg'])
    max_sliced_avg = np.array(results['max_sliced_avg'])
    avg_sliced_avg = np.array(results['avg_sliced_avg'])
    d_f1_max = np.array(results['d_f1_max'])
    d_f2_max = np.array(results['d_f2_max'])
    d_tildef2_max = np.array(results['d_tildef2_max'])
    max_sliced_max = np.array(results['max_sliced_max'])
    avg_sliced_max = np.array(results['avg_sliced_max'])
    d_f1_min = np.array(results['d_f1_min'])
    d_f2_min = np.array(results['d_f2_min'])
    d_tildef2_min = np.array(results['d_tildef2_min'])
    max_sliced_min = np.array(results['max_sliced_min'])
    avg_sliced_min = np.array(results['avg_sliced_min'])
    
    plt.semilogy(d_axis, d_f1_avg, label='$F_1$ IPM')
    plt.semilogy(d_axis, d_f2_avg, label='$F_2$ IPM')
    plt.semilogy(d_axis, d_tildef2_avg, label='$tilde{F}_2$ IPM')
    plt.semilogy(d_axis, max_sliced_avg, label='Max-sliced W.')
    plt.semilogy(d_axis, avg_sliced_avg, label='Sliced W.')
    plt.fill_between(d_axis, d_f1_min, d_f1_max, alpha=.3)
    plt.fill_between(d_axis, d_f2_min, d_f2_max, alpha=.3)
    plt.fill_between(d_axis, d_tildef2_min, d_tildef2_max, alpha=.3)
    plt.fill_between(d_axis, max_sliced_min, max_sliced_max, alpha=.3)
    plt.fill_between(d_axis, avg_sliced_min, avg_sliced_max, alpha=.3)
    
    plt.ylabel('Distances')
    plt.xlabel('Dimension d')
    plt.legend()
    
def fig_2_f1_f2_sliced_w(args, results):
    d_axis = np.array(args.d_vec)
    d_f1_nu_avg = np.array(results['d_f1_nu_avg'])
    d_f2_nu_avg = np.array(results['d_f2_nu_avg'])
    d_tildef2_nu_avg = np.array(results['d_tildef2_nu_avg'])
    max_sliced_nu_avg = np.array(results['max_sliced_nu_avg'])
    avg_sliced_nu_avg = np.array(results['avg_sliced_nu_avg'])
    d_f1_nu_max = np.array(results['d_f1_nu_max'])
    d_f2_nu_max = np.array(results['d_f2_nu_max'])
    d_tildef2_nu_max = np.array(results['d_tildef2_nu_max'])
    max_sliced_nu_max = np.array(results['max_sliced_nu_max'])
    avg_sliced_nu_max = np.array(results['avg_sliced_nu_max'])
    d_f1_nu_min = np.array(results['d_f1_nu_min'])
    d_f2_nu_min = np.array(results['d_f2_nu_min'])
    d_tildef2_nu_min = np.array(results['d_tildef2_nu_min'])
    max_sliced_nu_min = np.array(results['max_sliced_nu_min'])
    avg_sliced_nu_min = np.array(results['avg_sliced_nu_min'])
    
    plt.semilogy(d_axis, d_f1_nu_avg, label='$F_1$ IPM')
    plt.semilogy(d_axis, d_f2_nu_avg, label='$F_2$ IPM')
    plt.semilogy(d_axis, d_tildef2_nu_avg, label='$tilde{F}_2$ IPM')
    plt.semilogy(d_axis, max_sliced_nu_avg, label='Max-sliced W.')
    plt.semilogy(d_axis, avg_sliced_nu_avg, label='Sliced W.')
    plt.fill_between(d_axis, d_f1_nu_min, d_f1_nu_max, alpha=.3)
    plt.fill_between(d_axis, d_f2_nu_min, d_f2_nu_max, alpha=.3)
    plt.fill_between(d_axis, d_tildef2_nu_min, d_tildef2_nu_max, alpha=.3)
    plt.fill_between(d_axis, max_sliced_nu_min, max_sliced_nu_max, alpha=.3)
    plt.fill_between(d_axis, avg_sliced_nu_min, avg_sliced_nu_max, alpha=.3)
    
    plt.ylabel('Distances')
    plt.xlabel('Dimension d')
    plt.legend()
    
if __name__ == '__main__':
    
    args = empty_class()
    args.name = 'f1_f2_ipm'
    args.d_vec = [6,8,10,12,14,16]
    args.k = 6
    args.n_feature_samples = 10000
    args.n_samples = 40000000
    args.alpha = 1
    args.gamma = 1.0
    args.a = 1.0
    args.b = 0.0
    results = values_f1_f2_sep(args)
    plt.figure(figsize=(4,3))
    fig_1_f1_f2_sep(args, results)
    plt.title(f'$F_1$ and $F_2$ IPM estimates ($k = {args.k}$)')
    plt.savefig(f'figures/f1_f2_ipm_fig1_{args.k}_{args.n_samples}.pdf', bbox_inches='tight', pad_inches=0)
    plt.figure(figsize=(4,3))
    fig_2_f1_f2_sep(args, results)
    plt.title(f'Ratios between $F_1$ and $F_2$ IPM estimates ($k = {args.k}$)')
    plt.savefig(f'figures/f1_f2_ipm_fig2_{args.k}_{args.n_samples}.pdf', bbox_inches='tight', pad_inches=0)
    
    args = empty_class()
    args.name = 'f1_f2_sd'
    args.d_vec = [6,8,10,12,14,16]
    args.k = 5
    args.n_feature_samples = 10000
    args.n_samples = 500000
    args.alpha = 1
    args.gamma = 1.0
    args.a = 1.0
    args.b = 0.0
    results = values_f1_f2_sd(args)
    plt.figure(figsize=(4,3))
    fig_1_f1_f2_sd(args, results)
    plt.title(f'$F_1$ and $F_2$ SD estimates ($k = {args.k}$)')
    plt.savefig(f'figures/f1_f2_sd_fig1_{args.k}_{args.n_samples}.pdf', bbox_inches='tight', pad_inches=0)
    plt.figure(figsize=(4,3))
    fig_2_f1_f2_sd(args, results)
    plt.title(f'Ratios between $F_1$ and $F_2$ SD estimates ($k = {args.k}$)')
    plt.savefig(f'figures/f1_f2_sd_fig2_{args.k}_{args.n_samples}.pdf', bbox_inches='tight', pad_inches=0)

    args = empty_class()
    args.name = 'f1_f2_sliced_w'
    args.d_vec = [6,8,10,12,14,16]
    args.n_feature_samples = 10000
    args.n_samples = 100000
    args.alpha = 1
    args.large_var = 1
    args.small_var = 0.1
    args.a = 1.0
    args.b = 0.0
    results = values_f1_f2_sliced_w(args)
    plt.figure(figsize=(4,3))
    fig_1_f1_f2_sliced_w(args, results)
    plt.title(f'$F_1$ and $F_2$ IPM, sliced and max-sliced Wasserstein')
    plt.savefig(f'figures/f1_f2_sliced_w_fig1_{args.large_var}_{args.small_var}_{args.n_samples}.pdf', bbox_inches='tight', pad_inches=0)
    plt.figure(figsize=(4,3))
    fig_2_f1_f2_sliced_w(args, results)
    plt.title(f'$F_1$ and $F_2$ IPM, sliced and max-sliced Wasserstein (baseline)')
    plt.savefig(f'figures/f1_f2_sliced_w_fig2_{args.large_var}_{args.small_var}_{args.n_samples}.pdf', bbox_inches='tight', pad_inches=0)
