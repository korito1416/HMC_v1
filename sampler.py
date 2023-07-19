#!/usr/bin/env python

# Import Required Packages
# ========================
import os, sys
import pickle
import time

import matplotlib.pyplot as plt

# Import the solvers
import solvers



########################################################################
import argparse
parser = argparse.ArgumentParser(description="parameter settings")
parser.add_argument("--weight",type=float,default=0.25)
parser.add_argument("--xi",type=float,default=0.01)
parser.add_argument("--pf",type=float,default=20.76)
parser.add_argument("--pa",type=float,default=44.75)
parser.add_argument("--theta",type=float,default=1.0)
parser.add_argument("--gamma",type=float,default=1.0)
parser.add_argument("--sitenum",type=int,default=10)
parser.add_argument("--time",type=int,default=200)
parser.add_argument("--dataname",type=str,default="tests")
parser.add_argument("--mix_in",type=int,default=2)
parser.add_argument("--mass_matrix_theta",type=float,default=1)
parser.add_argument("--mass_matrix_gamma",type=float,default=1.0)
parser.add_argument("--symplectic_integrator_num_steps",type=int,default=2)


args = parser.parse_args()
weight = args.weight
pf = args.pf
pa = args.pa
theta_multiplier = args.theta
gamma_multiplier = args.gamma
sitenum = args.sitenum
time = args.time
xi = args.xi
dataname = args.dataname
mix_in= args.mix_in
mass_matrix_theta=args.mass_matrix_theta
mass_matrix_gamma=args.mass_matrix_gamma
symplectic_integrator_num_steps=args.symplectic_integrator_num_steps

workdir = os.getcwd()
output_dir = workdir+"/output/"+dataname+"/pf_"+str(pf)+"_pa_"+str(pa)+"_time_"+str(time)+"/theta_"+str(theta_multiplier)+"_gamma_"+str(gamma_multiplier)+"/sitenum_"+str(sitenum)+"_xi_"+str(xi)+"/mix_in_"+str(mix_in)+"_mass_matrix_theta_"+str(mass_matrix_theta)+"_mass_matrix_gamma_"+str(mass_matrix_gamma)+"_symplectic_integrator_num_steps_"+str(symplectic_integrator_num_steps)+"/weight_"+str(weight)+"/"
plotdir = workdir+"/plot/"+dataname+"/pf_"+str(pf)+"_pa_"+str(pa)+"_time_"+str(time)+"/theta_"+str(theta_multiplier)+"_gamma_"+str(gamma_multiplier)+"/sitenum_"+str(sitenum)+"_xi_"+str(xi)+"/mix_in_"+str(mix_in)+"_mass_matrix_theta_"+str(mass_matrix_theta)+"_mass_matrix_gamma_"+str(mass_matrix_gamma)+"_symplectic_integrator_num_steps_"+str(symplectic_integrator_num_steps)+"/weight_"+str(weight)+"/"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
if not os.path.exists(plotdir):
    os.makedirs(plotdir)

########################################################################

# results = mcmc_with_gams.main(weight = weight,
#                                        xi = xi,
#                                        pf=pf,
#                                        pa=pa,
#                                        site_num=sitenum,
#                                        T=time,
#                                        output_dir=output_dir,
                                    #    )
casadi_results = solvers.solve_with_casadi(weight = weight,
                                       xi = xi,
                                       pf=pf,
                                       pa=pa,
                                       site_num=sitenum,
                                       T=time,
                                       output_dir=output_dir,
                                       mix_in=mix_in,
                                       mass_matrix_theta=mass_matrix_theta,
                                       mass_matrix_gamma=mass_matrix_gamma,
                                       symplectic_integrator_num_steps=symplectic_integrator_num_steps,
                                       two_param_uncertainty = True,
                                       )
########################################################################
