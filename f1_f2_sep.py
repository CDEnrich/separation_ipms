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
    
def get_nu_samples(args, second_dataset=False):
    fname = os.path.join('nu_samples',
                         f'nu_samples_{args.d}_{args.k}_{args.n_samples}_{args.seed}.pkl')
    if os.path.exists(fname):
        X = pickle.load(open(fname, 'rb'))
        return X
    else:
        q_k_d = ss.jacobi(args.k, (args.d-3)/2.0, (args.d-3)/2.0)
        legendre_k_d = q_k_d/q_k_d(1)
        if not second_dataset:
            torch.manual_seed(args.seed)
        else:
            torch.manual_seed(10*args.seed)
        for j in range(args.n_samples//1000000):
            start = time.time()
            X0 = torch.randn(1000000,args.d)
            X0 = torch.nn.functional.normalize(X0, p=2, dim=1)
            acceptance_prob = torch.nn.functional.relu(torch.from_numpy(0.99*legendre_k_d(X0[:,args.d-1])))
            acceptance_vector = torch.bernoulli(acceptance_prob)
            accepted_rows = []
            for i in range(1000000):
                if acceptance_vector[i] == 1:
                    accepted_rows.append(i)
            accepted_rows_tensor = torch.tensor(accepted_rows).unsqueeze(1).expand([len(accepted_rows),args.d])
            if j==0:
                X = torch.gather(X0, 0, accepted_rows_tensor)
            else:
                samples = torch.gather(X0, 0, accepted_rows_tensor)
                X = torch.cat((X,samples),0)
                print(f'Sample batch {j+1}/{args.n_samples//1000000} done in {time.time()-start}. {samples.shape[0]} more samples.')
        if not os.path.exists('nu_samples'):
            os.makedirs('nu_samples')
        pickle.dump(X, open(fname, 'wb'))
        return X
    
def get_mu_samples(args):
    fname = os.path.join('mu_samples', 
                         f'mu_samples_{args.d}_{args.k}_{args.n_samples}_{args.seed}.pkl')
    if os.path.exists(fname):
        X = pickle.load(open(fname, 'rb'))
        return X
    else:
        q_k_d = ss.jacobi(args.k, (args.d-3)/2.0, (args.d-3)/2.0)
        legendre_k_d = q_k_d/q_k_d(1)
        torch.manual_seed(args.seed)
        for j in range(args.n_samples//1000000):
            X0 = torch.randn(1000000,args.d)
            X0 = torch.nn.functional.normalize(X0, p=2, dim=1)
            acceptance_prob = torch.nn.functional.relu(torch.from_numpy(-0.99*legendre_k_d(X0[:,args.d-1])))
            acceptance_vector = torch.bernoulli(acceptance_prob)
            accepted_rows = []
            for i in range(1000000):
                if acceptance_vector[i] == 1:
                    accepted_rows.append(i)
            accepted_rows_tensor = torch.tensor(accepted_rows).unsqueeze(1).expand([len(accepted_rows),args.d])
            if j==0:
                X = torch.gather(X0, 0, accepted_rows_tensor)
            else:
                samples = torch.gather(X0, 0, accepted_rows_tensor)
                X = torch.cat((X,samples),0)
                print(f'Sample batch {j+1}/{args.n_samples//1000000} done in {time.time()-start}. {samples.shape[0]} more samples.')
        if not os.path.exists('mu_samples'):
            os.makedirs('mu_samples')
        pickle.dump(X, open(fname, 'wb'))
        return X

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='F1/F2 IPM separation in sphere')
    parser.add_argument('--name', default='f1_f2_ipm', help='experiment name')
    parser.add_argument('--use_grid', action='store_true', help='use grid')
    parser.add_argument('--d', type=int, default=12, help='dimension of the data')
    parser.add_argument('--k', type=int, default=6, help='degree of Legendre polynomial')
    parser.add_argument('--n_samples', type=int, default=100000000, help='number of samples')
    parser.add_argument('--n_feature_samples', type=int, default=10000, help='number of feature samples')
    parser.add_argument('--seed', type=int, default=42, help='seed')
    parser.add_argument('--alpha', type=int, default=1, help='parameter of the activation function')
    parser.add_argument('--gamma', type=float, default=0.5, help='energy multiplier')
    parser.add_argument('--a', type=float, default=1.0, help='parameter of the activation function')
    parser.add_argument('--b', type=float, default=0.0, help='parameter of the activation function')
    parser.add_argument('--interactive', action='store_true', help='interactive, i.e. do not save results')
    parser.add_argument('--theoretical_f2', action='store_true', help='compute f2 distance with exact kernel too')
    
    args = parser.parse_args()
    
    def d_f1_estimate(X_nu, X_mu, args):
        gen_moment_nu_positive = args.a*torch.mean(torch.nn.functional.relu(X_nu[:,args.d-1])) + \
        args.b*torch.mean(torch.nn.functional.relu(-X_nu[:,args.d-1]))
        gen_moment_nu_negative = args.a*torch.mean(torch.nn.functional.relu(-X_nu[:,args.d-1])) + \
        args.b*torch.mean(torch.nn.functional.relu(X_nu[:,args.d-1]))
        gen_moment_mu_positive = args.a*torch.mean(torch.nn.functional.relu(X_mu[:,args.d-1])) + \
        args.b*torch.mean(torch.nn.functional.relu(-X_mu[:,args.d-1]))
        gen_moment_mu_negative = args.a*torch.mean(torch.nn.functional.relu(-X_mu[:,args.d-1])) + \
        args.b*torch.mean(torch.nn.functional.relu(X_mu[:,args.d-1]))
        return torch.max(torch.abs(gen_moment_nu_positive - gen_moment_mu_positive), 
                         torch.abs(gen_moment_nu_negative - gen_moment_mu_negative))
    
    def d_f2_estimate(X_nu, X_mu, args):
        torch.manual_seed(args.seed)
        Y0 = torch.randn(args.d,args.n_feature_samples)
        Y0 = torch.nn.functional.normalize(Y0, p=2, dim=0)
        gen_moment_nu_positive = args.a*torch.mean(torch.nn.functional.relu(torch.matmul(X_nu,Y0)), dim=0) + \
        args.b*torch.mean(torch.nn.functional.relu(-torch.matmul(X_nu,Y0)), dim=0)
        gen_moment_nu_negative = args.a*torch.mean(torch.nn.functional.relu(-torch.matmul(X_nu,Y0)), dim=0) + \
        args.b*torch.mean(torch.nn.functional.relu(torch.matmul(X_nu,Y0)), dim=0)
        gen_moment_mu_positive = args.a*torch.mean(torch.nn.functional.relu(torch.matmul(X_mu,Y0)), dim=0) + \
        args.b*torch.mean(torch.nn.functional.relu(-torch.matmul(X_mu,Y0)), dim=0)
        gen_moment_mu_negative = args.a*torch.mean(torch.nn.functional.relu(-torch.matmul(X_mu,Y0)), dim=0) + \
        args.b*torch.mean(torch.nn.functional.relu(torch.matmul(X_mu,Y0)), dim=0)
        d_f2_sq = torch.mean(0.5*(gen_moment_nu_positive-gen_moment_mu_positive)**2 + \
                             0.5*(gen_moment_nu_negative-gen_moment_mu_negative)**2)
        return torch.sqrt(d_f2_sq)
    
    def f2_kernel_evaluation(X0, X1, a, b, fill_diag = True):
        if fill_diag:
            inner_prod = torch.matmul(X0,X1.t()).fill_diagonal_(fill_value = 1)
        else:
            inner_prod = torch.matmul(X0,X1.t())
        values = (a+b)*((np.pi-torch.acos(inner_prod))*inner_prod \
                  + torch.sqrt(1-inner_prod*inner_prod))/(2*np.pi*(args.d+1))
        return values
    
    def d_f2_estimate_exact_kernel(X_nu, X_mu, a, b):
        kernel_eval_X_mu_X_mu = f2_kernel_evaluation(X_mu, X_mu, a, b)
        kernel_eval_X_nu_X_nu = f2_kernel_evaluation(X_nu, X_nu, a, b)
        kernel_eval_X_mu_X_nu = f2_kernel_evaluation(X_mu, X_nu, a, b, fill_diag = False)
        return np.sqrt(torch.mean(kernel_eval_X_mu_X_mu) + torch.mean(kernel_eval_X_nu_X_nu) - \
                       2*torch.mean(kernel_eval_X_mu_X_nu))
    
    def d_f1_estimate_theoretical(args):
        torch.manual_seed(args.seed)
        X0 = torch.randn(args.n_samples,args.d)
        X0 = torch.nn.functional.normalize(X0, p=2, dim=1)
        q_k_d = ss.jacobi(args.k, (args.d-3)/2.0, (args.d-3)/2.0)
        legendre_k_d = q_k_d/q_k_d(1)
        acceptance_prob_plus = torch.nn.functional.relu(torch.from_numpy(0.99*legendre_k_d(X0[:,args.d-1])))
        acceptance_prob_minus = torch.nn.functional.relu(torch.from_numpy(-0.99*legendre_k_d(X0[:,args.d-1])))
        A = torch.nn.functional.relu(X0[:,args.d-1])
        B = torch.nn.functional.relu(-X0[:,args.d-1])
        return torch.abs(2*torch.sum((args.a*A + args.b*B)*(acceptance_prob_plus-acceptance_prob_minus)))/ \
    torch.sum(acceptance_prob_plus+acceptance_prob_minus)
    
    def compute_distances(args, fname):
        start = time.time()
        X_nu = get_nu_samples(args)
        print(f'X_nu samples done. Duration={time.time()-start}')
        start = time.time()
        X_nu_2 = get_nu_samples(args, second_dataset=True)
        print(f'X_nu samples done. Duration={time.time()-start}')
        start = time.time()
        X_mu = get_mu_samples(args)
        print(f'X_mu samples done. Duration={time.time()-start}')
        min_num = np.min([X_nu.shape[0],X_nu_2.shape[0],X_mu.shape[0]])
        X_nu = X_nu[:(min_num),:]
        X_nu_2 = X_nu_2[:(min_num),:]
        X_mu = X_mu[:(min_num),:]
        print(f'Size of X_mu and X_nu: {X_nu.shape[0]}')
        #Distance estimates between nu and mu
        start = time.time()
        d_f1 = d_f1_estimate(X_nu, X_mu, args)
        print('D_{B_F1} estimate', float(d_f1))
        print(f'd={args.d}, k={args.k}, n_samples={args.n_samples}, duration={time.time()-start}')
        start = time.time()
        d_f2 = d_f2_estimate(X_nu, X_mu, args)
        print('D_{B_F2} estimate', float(d_f2))
        print(f'd={args.d}, k={args.k}, n_samples={args.n_samples}, n_feature_samples={args.n_feature_samples}, duration={time.time()-start}')
        start = time.time()
        d_f1_t = d_f1_estimate_theoretical(args)
        print('D_{B_F1} theoretical estimate', float(d_f1_t))
        print(f'd={args.d}, k={args.k}, n_samples={args.n_samples}, duration={time.time()-start}')
        if args.theoretical_f2:
            start = time.time()
            d_f2_t = d_f2_estimate_exact_kernel(X_nu, X_mu, args.a, args.b)
            print('D_{B_F2} theoretical estimate', float(d_f2_t))
            print(f'd={args.d}, k={args.k}, n_samples={args.n_samples}, duration={time.time()-start}')
        #Distance estimates between nu and itself
        start = time.time()
        d_f1_nu = d_f1_estimate(X_nu, X_nu_2, args)
        print('D_{B_F1} estimate between nu and itself', float(d_f1_nu))
        print(f'd={args.d}, k={args.k}, n_samples={args.n_samples}, duration={time.time()-start}')
        start = time.time()
        d_f2_nu = d_f2_estimate(X_nu, X_nu_2, args)
        print('D_{B_F2} estimate between nu and itself', float(d_f2_nu))
        print(f'd={args.d}, k={args.k}, n_samples={args.n_samples}, n_feature_samples={args.n_feature_samples}, duration={time.time()-start}')
        N_kd = (2*args.k + args.d - 2) * math.factorial(args.k + args.d - 3) / (math.factorial(args.k) * math.factorial(args.d -2))
        print(f'dF1/dF2 ratio: {(d_f1+d_f1_t)/(2*d_f2)}. sqrt(N_kd): {np.sqrt(N_kd)}')
        if args.theoretical_f2:
            res = {
                'd_f1': d_f1,
                'd_f2': d_f2,
                'd_f1_t': d_f1_t,
                'd_f2_t': d_f2_t,
                'd_f1_nu': d_f1_nu,
                'd_f2_nu': d_f2_nu,
                'ratio': (d_f1+d_f1_t)/(2*d_f2),
                'sqrt(N_kd)': np.sqrt(N_kd),
            }
        else:
            res = {
                'd_f1': d_f1,
                'd_f2': d_f2,
                'd_f1_t': d_f1_t,
                'd_f1_nu': d_f1_nu,
                'd_f2_nu': d_f2_nu,
                'ratio': (d_f1+d_f1_t)/(2*d_f2),
                'sqrt(N_kd)': np.sqrt(N_kd),
            }
        if not args.interactive:
            pickle.dump(res, open(fname, 'wb'))
    
    if not args.use_grid:
        resdir = os.path.join('res', args.name)
        if not os.path.exists(resdir):
            os.makedirs(resdir)
        fname = os.path.join(resdir,f'{args.name}_{args.d}_{args.k}_{args.n_samples}_{args.n_feature_samples}_{args.seed}_{args.alpha}_{args.gamma}_{args.a}_{args.b}.pkl')
        print(f'Output:{fname}')
        if os.path.exists(fname) and not args.interactive:
            print('results file already exists, skipping')
            sys.exit(0)
        compute_distances(args)
    else:
        d_vec = [6,8,10,12,14,16]
        resdir = os.path.join('res', args.name)
        if not os.path.exists(resdir):
            os.makedirs(resdir)
        for i in range(len(d_vec)):
            args.d = d_vec[i]
            fname = os.path.join(resdir,f'{args.name}_{args.d}_{args.k}_{args.n_samples}_{args.n_feature_samples}_{args.seed}_{args.alpha}_{args.gamma}_{args.a}_{args.b}.pkl')
            print(f'Output:{fname}')
            if os.path.exists(fname) and not args.interactive:
                print(f'results file already exists, skipping')
                continue
            print(f'Dimension {i+1}/{args.d}')
            compute_distances(args, fname)
    
    