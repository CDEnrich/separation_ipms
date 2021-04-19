import numpy as np
import time
import math
import os
import argparse
import pickle
import torch
import torch.nn as nn
import scipy.special as ss
import scipy.stats as sst

def set_args_for_task_id(args, task_id):
    grid = {
        'd': [6, 8, 10, 12, 14, 16],
        'seed': [42, 43, 44, 45, 46, 47, 48, 49, 50, 51],
    }
    from itertools import product
    gridlist = list(dict(zip(grid.keys(), vals)) for vals in product(*grid.values()))
    print(f'task {task_id} out of {len(gridlist)}')
    assert task_id >= 1 and task_id <= len(gridlist), 'wrong task_id!'
    elem = gridlist[task_id - 1]
    for k, v in elem.items():
        setattr(args, k, v)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='F1/F2 and sliced Wasserstein in Euclidean space')
    parser.add_argument('--name', default='f1_f2_sliced_w', help='experiment name')
    parser.add_argument('--use_grid', action='store_true', help='use grid')
    parser.add_argument('--d', type=int, default=12, help='dimension of the data')
    parser.add_argument('--n_samples', type=int, default=10000, help='number of samples')
    parser.add_argument('--n_feature_samples', type=int, default=10000, help='number of feature samples')
    parser.add_argument('--seed', type=int, default=42, help='seed')
    parser.add_argument('--alpha', type=int, default=1, help='parameter of the activation function')
    parser.add_argument('--a', type=float, default=1.0, help='parameter of the activation function')
    parser.add_argument('--b', type=float, default=0.0, help='parameter of the activation function')
    parser.add_argument('--large_var', type=float, default=1, help='large variance')
    parser.add_argument('--small_var', type=float, default=0.1, help='small variance')
    parser.add_argument('--interactive', action='store_true', help='interactive, i.e. do not save results')
    parser.add_argument('--task_id', type=int, default=None, help='task id for sweep jobs')

    args = parser.parse_args()
    
    if args.task_id is not None:
        set_args_for_task_id(args, args.task_id)

    def max_sliced(X_nu, X_mu, args):
        return sst.wasserstein_distance(X_mu[:,args.d-1], X_nu[:,args.d-1])

    def avg_sliced(X_nu, X_mu, args):
        torch.manual_seed(args.seed)
        Y0 = torch.randn(args.d,args.n_feature_samples)
        Y0 = torch.nn.functional.normalize(Y0, p=2, dim=0).double()
        average = 0
        for i in range(args.n_feature_samples):
            average = average + sst.wasserstein_distance(torch.matmul(torch.from_numpy(X_mu),Y0[:,i]),
                                                         torch.matmul(torch.from_numpy(X_nu),Y0[:,i]))
        average = average/args.n_feature_samples
        return average
    
    def d_f1_estimate_w(X_nu, X_mu, args):
        ones_d = torch.ones(args.n_samples,1).double()
        X_mu = torch.cat((torch.from_numpy(X_mu),ones_d), 1)
        X_nu = torch.cat((torch.from_numpy(X_nu),ones_d), 1)
        gen_moment_nu_positive = args.a*torch.mean(torch.nn.functional.relu(torch.sqrt(X_nu[:,args.d-1]**2 + 1))) + args.b*torch.mean(torch.nn.functional.relu(-torch.sqrt(X_nu[:,args.d-1]**2 + 1)))
        gen_moment_nu_negative = args.a*torch.mean(torch.nn.functional.relu(-torch.sqrt(X_nu[:,args.d-1]**2 + 1))) + args.b*torch.mean(torch.nn.functional.relu(torch.sqrt(X_nu[:,args.d-1]**2 + 1)))
        gen_moment_mu_positive = args.a*torch.mean(torch.nn.functional.relu(torch.sqrt(X_mu[:,args.d-1]**2 + 1))) + args.b*torch.mean(torch.nn.functional.relu(-torch.sqrt(X_mu[:,args.d-1]**2 + 1)))
        gen_moment_mu_negative = args.a*torch.mean(torch.nn.functional.relu(-torch.sqrt(X_mu[:,args.d-1]**2 + 1))) + args.b*torch.mean(torch.nn.functional.relu(torch.sqrt(X_mu[:,args.d-1]**2 + 1)))
        return torch.max(torch.abs(gen_moment_nu_positive - gen_moment_mu_positive),torch.abs(gen_moment_nu_negative - gen_moment_mu_negative))
    
    def d_f2_estimate_w(X_nu, X_mu, args):
        ones_d = torch.ones(args.n_samples,1).double()
        X_mu = torch.cat((torch.from_numpy(X_mu),ones_d), 1)
        X_nu = torch.cat((torch.from_numpy(X_nu),ones_d), 1)
        torch.manual_seed(args.seed)
        Y0 = torch.randn(args.d+1,args.n_feature_samples)
        Y0 = torch.nn.functional.normalize(Y0, p=2, dim=0).double()
        gen_moment_nu_positive = args.a*torch.mean(torch.nn.functional.relu(torch.matmul(X_nu,Y0)), dim=0) + args.b*torch.mean(torch.nn.functional.relu(-torch.matmul(X_nu,Y0)), dim=0)
        gen_moment_nu_negative = args.a*torch.mean(torch.nn.functional.relu(-torch.matmul(X_nu,Y0)), dim=0) + args.b*torch.mean(torch.nn.functional.relu(torch.matmul(X_nu,Y0)), dim=0)
        gen_moment_mu_positive = args.a*torch.mean(torch.nn.functional.relu(torch.matmul(X_mu,Y0)), dim=0) + args.b*torch.mean(torch.nn.functional.relu(-torch.matmul(X_mu,Y0)), dim=0)
        gen_moment_mu_negative = args.a*torch.mean(torch.nn.functional.relu(-torch.matmul(X_mu,Y0)), dim=0) + args.b*torch.mean(torch.nn.functional.relu(torch.matmul(X_mu,Y0)), dim=0)
        d_f2_sq = torch.mean(0.5*(gen_moment_nu_positive-gen_moment_mu_positive)**2 + 0.5*(gen_moment_nu_negative-gen_moment_mu_negative)**2)
        return torch.sqrt(d_f2_sq)
    
    def d_tilde_f2_estimate_w(X_nu, X_mu, args):
        ones_d = torch.ones(args.n_samples,1).double()
        X_mu = torch.cat((torch.from_numpy(X_mu),ones_d), 1)
        X_nu = torch.cat((torch.from_numpy(X_nu),ones_d), 1)
        torch.manual_seed(args.seed)
        Z0 = torch.randn(args.d,args.n_feature_samples)
        Z0 = torch.nn.functional.normalize(Z0, p=2, dim=0).double()
        w0 = (np.pi*torch.rand(args.n_feature_samples) - 0.5*np.pi).unsqueeze(0).double()
        Y0 = torch.cat((torch.cos(w0)*Z0,torch.sin(w0)),0)
        gen_moment_nu_positive = args.a*torch.mean(torch.nn.functional.relu(torch.matmul(X_nu,Y0)), dim=0) + args.b*torch.mean(torch.nn.functional.relu(-torch.matmul(X_nu,Y0)), dim=0)
        gen_moment_nu_negative = args.a*torch.mean(torch.nn.functional.relu(-torch.matmul(X_nu,Y0)), dim=0) + args.b*torch.mean(torch.nn.functional.relu(torch.matmul(X_nu,Y0)), dim=0)
        gen_moment_mu_positive = args.a*torch.mean(torch.nn.functional.relu(torch.matmul(X_mu,Y0)), dim=0) + args.b*torch.mean(torch.nn.functional.relu(-torch.matmul(X_mu,Y0)), dim=0)
        gen_moment_mu_negative = args.a*torch.mean(torch.nn.functional.relu(-torch.matmul(X_mu,Y0)), dim=0) + args.b*torch.mean(torch.nn.functional.relu(torch.matmul(X_mu,Y0)), dim=0)
        d_f2_sq = torch.mean(0.5*(gen_moment_nu_positive-gen_moment_mu_positive)**2 + 0.5*(gen_moment_nu_negative-gen_moment_mu_negative)**2)
        return torch.sqrt(d_f2_sq)
    
    def compute_distances(args, fname):
        mu_variance = args.large_var*np.identity(args.d)
        nu_variance = args.large_var*np.identity(args.d)
        nu_variance[args.d-1, args.d-1] = args.small_var
        torch.manual_seed(args.seed)
        X_mu = np.random.multivariate_normal(np.zeros(args.d), mu_variance, args.n_samples)
        X_nu = np.random.multivariate_normal(np.zeros(args.d), nu_variance, args.n_samples)
        torch.manual_seed(10*args.seed)
        X_nu_2 = np.random.multivariate_normal(np.zeros(args.d), nu_variance, args.n_samples)
        torch.manual_seed(args.seed)
        print(f'Size of X_mu and X_nu: {X_nu.shape[0]}')
        #Distance estimates between nu and mu
        start = time.time()
        d_f1 = d_f1_estimate_w(X_nu, X_mu, args)
        print('D_{B_F1} estimate', float(d_f1))
        print(f'd={args.d}, n_samples={args.n_samples}, duration={time.time()-start}')
        start = time.time()
        max_sl = max_sliced(X_nu, X_mu, args)
        print('Max sliced Wasserstein estimate', float(max_sl))
        print(f'd={args.d}, n_samples={args.n_samples}, duration={time.time()-start}')
        start = time.time()
        d_f2 = d_f2_estimate_w(X_nu, X_mu, args)
        print('D_{B_F2} estimate', float(d_f2))
        print(f'd={args.d}, n_samples={args.n_samples}, n_feature_samples={args.n_feature_samples}, duration={time.time()-start}')
        start = time.time()
        d_tildef2 = d_tilde_f2_estimate_w(X_nu, X_mu, args)
        print('D_{B_tildeF2} estimate', float(d_tildef2))
        print(f'd={args.d}, n_samples={args.n_samples}, n_feature_samples={args.n_feature_samples}, duration={time.time()-start}')
        start = time.time()
        avg_sl = avg_sliced(X_nu, X_mu, args)
        print('Avg sliced Wasserstein estimate', float(avg_sl))
        print(f'd={args.d}, n_samples={args.n_samples}, duration={time.time()-start}')
        #Distance estimates between nu and itself
        start = time.time()
        d_f1_nu = d_f1_estimate_w(X_nu, X_nu_2, args)
        print('D_{B_F1} estimate nu and itself', float(d_f1_nu))
        print(f'd={args.d}, n_samples={args.n_samples}, duration={time.time()-start}')
        start = time.time()
        max_sl_nu = max_sliced(X_nu, X_nu_2, args)
        print('Max sliced Wasserstein estimate nu and itself', float(max_sl_nu))
        print(f'd={args.d}, n_samples={args.n_samples}, duration={time.time()-start}')
        start = time.time()
        d_f2_nu = d_f2_estimate_w(X_nu, X_nu_2, args)
        print('D_{B_F2} estimate nu and itself', float(d_f2_nu))
        print(f'd={args.d}, n_samples={args.n_samples}, n_feature_samples={args.n_feature_samples}, duration={time.time()-start}')
        start = time.time()
        d_tildef2_nu = d_tilde_f2_estimate_w(X_nu, X_nu_2, args)
        print('D_{B_tildeF2} estimate nu and itself', float(d_tildef2_nu))
        print(f'd={args.d}, n_samples={args.n_samples}, n_feature_samples={args.n_feature_samples}, duration={time.time()-start}')
        start = time.time()
        avg_sl_nu = avg_sliced(X_nu, X_nu_2, args)
        print('Avg sliced Wasserstein estimate nu and itself', float(avg_sl_nu))
        print(f'd={args.d}, n_samples={args.n_samples}, duration={time.time()-start}')
        res = {
            'd_f1': d_f1,
            'max_sliced': max_sl,
            'd_f2': d_f2,
            'd_tildef2': d_tildef2,
            'avg_sliced': avg_sl,
            'd_f1_nu': d_f1_nu,
            'max_sliced_nu': max_sl_nu,
            'd_f2_nu': d_f2_nu,
            'd_tildef2_nu': d_tildef2_nu,
            'avg_sliced_nu': avg_sl_nu,
        }
        if not args.interactive:
            pickle.dump(res, open(fname, 'wb'))
    
    if args.task_id is not None or args.use_grid is not None:
        resdir = os.path.join('res', args.name)
        if not os.path.exists(resdir):
            os.makedirs(resdir)
        fname = os.path.join(resdir,f'{args.name}_{args.d}_{args.n_samples}_{args.n_feature_samples}_{args.seed}_{args.alpha}_{args.large_var}_{args.small_var}.pkl')
        print(f'Output:{fname}')
        if os.path.exists(fname) and not args.interactive:
            print('results file already exists, skipping')
            sys.exit(0)
        compute_distances(args, fname)
    else:
        d_vec = [6,8,10,12,14,16]
        resdir = os.path.join('res', args.name)
        if not os.path.exists(resdir):
            os.makedirs(resdir)
        for i in range(len(d_vec)):
            args.d = d_vec[i]
            fname = os.path.join(resdir,f'{args.name}_{args.d}_{args.k}_{args.n_samples}_{args.n_feature_samples}_{args.seed}_{args.alpha}_{args.large_var}_{args.small_var}.pkl')
            print(f'Output:{fname}')
            if os.path.exists(fname) and not args.interactive:
                print(f'results file already exists, skipping')
                continue
            print(f'Dimension {i+1}/{args.d}')
            compute_distances(args, fname)
        
    
    
    
