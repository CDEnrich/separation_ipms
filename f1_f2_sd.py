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

def get_mu_samples_sd(args):
    fname = os.path.join('mu_samples_sd', 
                         f'mu_samples_sd_{args.d}_{args.k}_{args.n_samples}_{args.seed}.pkl')
    if os.path.exists(fname):
        X = pickle.load(open(fname, 'rb'))
        return X
    else:
        torch.manual_seed(args.seed)
        X = torch.randn(args.n_samples,args.d)
        X = torch.nn.functional.normalize(X, p=2, dim=1)
        if not os.path.exists('mu_samples_sd'):
            os.makedirs('mu_samples_sd')
        pickle.dump(X, open(fname, 'wb'))
    return X

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
    parser = argparse.ArgumentParser(description='F1/F2 SD separation in sphere')
    parser.add_argument('--name', default='f1_f2_sd', help='experiment name')
    parser.add_argument('--use_grid', action='store_true', help='use grid')
    parser.add_argument('--d', type=int, default=12, help='dimension of the data')
    parser.add_argument('--k', type=int, default=5, help='degree of Legendre polynomial')
    parser.add_argument('--n_samples', type=int, default=200000, help='number of samples')
    parser.add_argument('--n_feature_samples', type=int, default=10000, help='number of feature samples')
    parser.add_argument('--seed', type=int, default=42, help='seed')
    parser.add_argument('--alpha', type=int, default=1, help='parameter of the activation function')
    parser.add_argument('--gamma', type=float, default=1.0, help='energy multiplier')
    parser.add_argument('--a', type=float, default=1.0, help='parameter of the activation function')
    parser.add_argument('--b', type=float, default=0.0, help='parameter of the activation function')
    parser.add_argument('--interactive', action='store_true', help='interactive, i.e. do not save results')
    parser.add_argument('--no_sd_f2', action='store_true', help='do not compute sd_f2')
    parser.add_argument('--task_id', type=int, default=None, help='task id for sweep jobs')
    parser.add_argument('--recompute_sd_f1_e', action='store_true', help='compute again sd_f1_e')

    args = parser.parse_args()

    if args.task_id is not None:
        set_args_for_task_id(args, args.task_id)
    
    def sd_f1_estimate_theoretical(args):
        if (args.k%2 != (args.alpha+1)%2) and (args.k > args.alpha + 2):
            lambda_alpha_p1_k_d = math.gamma(args.d/2)*math.factorial(args.alpha + 1)*math.gamma((args.d-1)/2)* \
            math.gamma(args.k - args.alpha - 1)/(np.sqrt(np.pi)*math.gamma((args.d-1)/2)*(2**args.k)*math.gamma((args.k - args.alpha)/2) *math.gamma((args.k + args.d + args.alpha + 1)/2))
        else:
            lambda_alpha_p1_k_d = 0
        result = args.gamma*lambda_alpha_p1_k_d*args.k*(args.d+args.k-3)/(args.alpha+1)
        return result
    
    def lambda_alpha_p1_k_d(X_mu,args):
        lambda_alpha_p1_k_d = math.gamma(args.d/2)*((-1)**((args.k-args.alpha-2)/2))*math.factorial(args.alpha + 1)*math.gamma((args.d-1)/2)* \
            math.gamma(args.k - args.alpha - 1)/(np.sqrt(np.pi)*math.gamma((args.d-1)/2)*(2**args.k)*math.gamma((args.k - args.alpha)/2) *math.gamma((args.k + args.d + args.alpha + 1)/2))
        q_k_d = ss.jacobi(args.k, (args.d-3)/2.0, (args.d-3)/2.0)
        legendre_k_d = q_k_d/q_k_d(1)
        lambda_alpha_p1_k_d_empirical = torch.mean(torch.from_numpy(legendre_k_d(X_mu[:,args.d-1]))*torch.nn.functional.relu(X_mu[:,args.d-1])**2)
        lambda_alpha_kp1_d = math.gamma(args.d/2)*((-1)**((args.k-args.alpha)/2))*math.factorial(args.alpha)*math.gamma((args.d-1)/2)* \
            math.gamma(args.k +1 - args.alpha)/(np.sqrt(np.pi)*math.gamma((args.d-1)/2)*(2**(args.k+1))*math.gamma((args.k - args.alpha + 2)/2) *math.gamma((args.k + args.d + args.alpha + 1)/2))
        q_kp1_d = ss.jacobi(args.k + 1, (args.d-3)/2.0, (args.d-3)/2.0)
        legendre_kp1_d = q_kp1_d/q_kp1_d(1)
        lambda_alpha_kp1_d_empirical = torch.mean(torch.from_numpy(legendre_kp1_d(X_mu[:,args.d-1]))*torch.nn.functional.relu(X_mu[:,args.d-1]))
        print(f'Theoretical lambda alphap1: {lambda_alpha_p1_k_d}. Empirical lambda alphap1: {lambda_alpha_p1_k_d_empirical}')
        print(f'Theoretical lambda alphap1: {lambda_alpha_kp1_d}. Empirical lambda alphap1: {lambda_alpha_kp1_d_empirical}')
    
    def sd_ratio_lower_bound_theoretical(args):
        N_kd = (2*args.k + args.d - 2) * math.factorial(args.k + args.d - 3) / (math.factorial(args.k) * math.factorial(args.d -2))
        numerator = args.k*(args.d+args.k-3)/(args.alpha+1)
        denominator = np.sqrt(2*(args.k*(args.k + args.d - 2)*(args.d + args.alpha - 2)**2/(args.alpha + 1)**2 + numerator**2))
        return numerator/(denominator/np.sqrt(N_kd))
    
    def score_function(X):
        n_samples = X.shape[0]
        derivative_factor = args.k*(args.k + args.d - 2)/(args.d - 1)
        q_km1_dp2 = ss.jacobi(args.k-1, (args.d-1)/2.0, (args.d-1)/2.0)
        legendre_km1_dp2 = q_km1_dp2/q_km1_dp2(1)
        e_d = torch.zeros(1,args.d)
        e_d[0,args.d-1] = 1
        result = args.gamma*derivative_factor*torch.from_numpy(legendre_km1_dp2(X[:,args.d-1])).unsqueeze(1)*e_d.repeat(n_samples,1)
        result = result - torch.sum((X.squeeze(0)*result), dim=1).unsqueeze(1)*X.squeeze(0)
        return result
    
    def N_kd_inv(X_mu, args):
        N_kd = (2*args.k + args.d - 2) * math.factorial(args.k + args.d - 3) / (math.factorial(args.k) * math.factorial(args.d -2))
        N_kd_inv = 1/N_kd
        q_k_d = ss.jacobi(args.k, (args.d-3)/2.0, (args.d-3)/2.0)
        legendre_k_d = q_k_d/q_k_d(1)
        N_kd_inv_empirical = torch.mean(torch.from_numpy(legendre_k_d(X_mu[:,args.d-1]))**2)
        print(f'N_kd_inv: {N_kd_inv}, N_kd_inv_empirical: {N_kd_inv_empirical}')
        score = score_function(X_mu)
        grad_norm = torch.mean(torch.norm(score, p=2, dim=1)**2)
        print(f'Gradient norm empirical: {grad_norm}. Gradient norm theoretical: {N_kd_inv*args.k*(args.k + args.d - 2)}')
        numerator = args.k*(args.d+args.k-3)/(args.alpha+1)
        denominator = np.sqrt(2*(args.k*(args.k + args.d - 2)*(args.d + args.alpha - 2)**2/(args.alpha + 1)**2 + numerator**2))
        print(numerator, denominator, numerator/denominator, np.sqrt(N_kd))
    
    def sd_f2_estimate(X_mu, args):
        torch.manual_seed(args.seed)
        d_f2_sq = 0
        mu_positive = torch.zeros(args.n_feature_samples,args.d)
        mu_negative = torch.zeros(args.n_feature_samples,args.d)
        for j in range(X_mu.shape[0]//10000):
            Y0 = torch.randn(args.d,args.n_feature_samples)
            Y0 = torch.nn.functional.normalize(Y0, p=2, dim=0)
            X_mu_s = X_mu[j*10000:(j+1)*10000,:]
            score_mu = score_function(X_mu_s)
            mu_positive = mu_positive + args.a*torch.mean(torch.nn.functional.relu(torch.matmul(X_mu_s,Y0)).unsqueeze(2)*score_mu.unsqueeze(1), dim=0) + args.b*torch.mean(torch.nn.functional.relu(-torch.matmul(X_mu_s,Y0)).unsqueeze(2)*score_mu.unsqueeze(1), dim=0)
            mu_negative = mu_negative + args.a*torch.mean(torch.nn.functional.relu(-torch.matmul(X_mu_s,Y0)).unsqueeze(2)*score_mu.unsqueeze(1), dim=0) + args.b*torch.mean(torch.nn.functional.relu(torch.matmul(X_mu_s,Y0)).unsqueeze(2)*score_mu.unsqueeze(1), dim=0)
        mu_positive = mu_positive/(X_mu.shape[0]//10000)
        mu_negative = mu_negative/(X_mu.shape[0]//10000)
        d_f2_sq = torch.mean(0.5*torch.norm(mu_positive, dim=1, p=2)**2 + 0.5*torch.norm(mu_negative, dim=1, p=2)**2)
        return torch.sqrt(d_f2_sq)
    
    def theoretical_estimate_opti(args):
        q_k_d = ss.jacobi(args.k, (args.d-3)/2.0, (args.d-3)/2.0)
        legendre_k_d = q_k_d/q_k_d(1)
        q_km1_dp2 = ss.jacobi(args.k-1, (args.d-1)/2.0, (args.d-1)/2.0)
        legendre_km1_dp2 = q_km1_dp2/q_km1_dp2(1)
        t_values = torch.linspace(-1,1, steps=200001)
        objective_values = -(args.d + args.alpha -2)*(args.k + args.d - 2)/(args.d - 1)*t_values*torch.sqrt(1-t_values**2)*legendre_km1_dp2(t_values) + (args.d + args.k - 3)*torch.sqrt(1-t_values**2)*legendre_k_d(t_values)
        lambda_alpha_p1_k_d = math.gamma(args.d/2)*math.factorial(args.alpha + 1)*math.gamma((args.d-1)/2)* \
                            math.gamma(args.k - args.alpha - 1)/(np.sqrt(np.pi)*math.gamma((args.d-1)/2)*(2**args.k)*math.gamma((args.k - args.alpha)/2) *math.gamma((args.k + args.d + args.alpha + 1)/2))
        return args.k/(args.alpha + 1)*lambda_alpha_p1_k_d*torch.max(torch.abs(objective_values))
    
    def theoretical_estimate_opti_d(args):
        q_k_d = ss.jacobi(args.k, (args.d-3)/2.0, (args.d-3)/2.0)
        legendre_k_d = q_k_d/q_k_d(1)
        q_km1_dp2 = ss.jacobi(args.k-1, (args.d-1)/2.0, (args.d-1)/2.0)
        legendre_km1_dp2 = q_km1_dp2/q_km1_dp2(1)
        t_values = torch.linspace(-1,1, steps=200001)
        objective_values = (args.d + args.alpha -2)*(args.k + args.d - 2)/(args.d - 1)*(1-t_values**2)*legendre_km1_dp2(t_values) + (args.d + args.k - 3)*t_values*legendre_k_d(t_values)
        lambda_alpha_p1_k_d = math.gamma(args.d/2)*math.factorial(args.alpha + 1)*math.gamma((args.d-1)/2)* \
                            math.gamma(args.k - args.alpha - 1)/(np.sqrt(np.pi)*math.gamma((args.d-1)/2)*(2**args.k)*math.gamma((args.k - args.alpha)/2) *math.gamma((args.k + args.d + args.alpha + 1)/2))
        return args.k/(args.alpha + 1)*lambda_alpha_p1_k_d*torch.max(torch.abs(objective_values))
    
    def sd_f1_estimate(X_mu, args):
        return torch.sqrt(theoretical_estimate_opti_d(args)**2+(args.d-1)*theoretical_estimate_opti(args)**2)
    
    def compute_distances(args, fname):
        start = time.time()
        X_mu = get_mu_samples_sd(args)
        print(f'X_mu samples done. Duration={time.time()-start}')
        print(f'Size of X_mu: {X_mu.shape[0]}')
        start = time.time()
        sd_f1 = sd_f1_estimate(X_mu, args)
        print('SD_{B_F1} estimate', float(sd_f1))
        print(f'd={args.d}, k={args.k}, n_samples={args.n_samples}, duration={time.time()-start}')
        if not args.no_sd_f2:
            start = time.time()
            sd_f2 = sd_f2_estimate(X_mu, args)
            print('SD_{B_F2} estimate', float(sd_f2))
            print(f'd={args.d}, k={args.k}, n_samples={args.n_samples}, n_feature_samples={args.n_feature_samples}, duration={time.time()-start}')
        start = time.time()
        sd_f1_t = sd_f1_estimate_theoretical(args)
        print('SD_{B_F1} theoretical estimate', float(sd_f1_t))
        print(f'd={args.d}, k={args.k}, duration={time.time()-start}')
        start = time.time()
        sd_ratio_t = sd_ratio_lower_bound_theoretical(args)
        if not args.no_sd_f2:
            print('SD_{B_F1}/SD_{B_F2} ratio theoretical lower bound:', float(sd_ratio_t), 'Ratio estimate:', float((sd_f1+sd_f1_t)/(2*sd_f2)), 'Ratio estimate 1:', float(sd_f1/sd_f2), 'Ratio estimate 2:', float(sd_f1_t/sd_f2))
        else:
            print('SD_{B_F1}/SD_{B_F2} ratio theoretical lower bound:', float(sd_ratio_t))
        print(f'd={args.d}, k={args.k}, n_samples={args.n_samples}, duration={time.time()-start}')
        lambda_alpha_p1_k_d(X_mu,args)
        N_kd_inv(X_mu, args)
        if not args.no_sd_f2:
            res = {
                'sd_f1_e': sd_f1,
                'sd_f2': sd_f2,
                'sd_f1_t': sd_f1_t,
                'sd_ratio': sd_f1/sd_f2,
                'sd_ratio_t': sd_ratio_t,
            }
        else:
            res = {
                'sd_f1': sd_f1,
                'sd_f1_t': sd_f1_t,
                'sd_ratio_t': sd_ratio_t,
            }
        if not args.interactive:
            pickle.dump(res, open(fname, 'wb'))
        
    if args.task_id is not None or args.use_grid is not None:
        resdir = os.path.join('res', args.name)
        if not os.path.exists(resdir):
            os.makedirs(resdir)
        fname = os.path.join(resdir,f'{args.name}_{args.d}_{args.k}_{args.n_samples}_{args.n_feature_samples}_{args.seed}_{args.alpha}_{args.gamma}_{args.a}_{args.b}.pkl')
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
            fname = os.path.join(resdir,f'{args.name}_{args.d}_{args.k}_{args.n_samples}_{args.n_feature_samples}_{args.seed}_{args.alpha}_{args.gamma}_{args.a}_{args.b}.pkl')
            print(f'Output:{fname}')
            if os.path.exists(fname) and not args.interactive:
                print(f'results file already exists, skipping')
                continue
            print(f'Dimension {i+1}/{args.d}')
            compute_distances(args, fname)
    
