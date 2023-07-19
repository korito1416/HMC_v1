
sitenumarray=(25)
xiarray=(0.1)
pfarray=(20.76)
paarray=(44.75)
thetaarray=(1.0)
gammaarray=(1.0)
timearray=(200)
weightarray=(0.1)
mix_in_array=(2)
mass_matrix_theta_scale_array=(1.0)
mass_matrix_gamma_scale_array=(1.0)
mass_matrix_weight_array=(0.5)
symplectic_integrator_num_steps_array=(10 20)
stepsize_array=(0.1 0.05 0.01)
hmc_python_name="sampler.py"

for sitenum in "${sitenumarray[@]}"; do
    for xi in "${xiarray[@]}"; do
        for pf in "${pfarray[@]}"; do
            for pa in "${paarray[@]}"; do
                for time in "${timearray[@]}"; do
                    for weight in "${weightarray[@]}"; do
                        for theta in "${thetaarray[@]}"; do
                            for gamma in "${gammaarray[@]}"; do
                                for mix_in in "${mix_in_array[@]}"; do
                                    for mass_matrix_theta_scale in "${mass_matrix_theta_scale_array[@]}"; do
                                        for mass_matrix_gamma_scale in "${mass_matrix_gamma_scale_array[@]}"; do
                                            for mass_matrix_weight in "${mass_matrix_weight_array[@]}"; do
                                                for symplectic_integrator_num_steps in "${symplectic_integrator_num_steps_array[@]}"; do
                                                    for stepsize in "${stepsize_array[@]}"; do
                                                        count=0
                                                                    
                                                        action_name="test_gams_rej"
                                                        # action_name="test2_casadi_rej"
                                                        # action_name="test2_casadi_results"
                                                        # action_name="test_gams_results"
                                                        action_name="casadi_massmatrix1"
                                                        action_name="casadi_massmatrix_nocons"
                                                        action_name="test_scale"
                                                        action_name="new_Test_1"


                                                        dataname="${action_name}"

                                                        mkdir -p ./job-outs/${action_name}/pf_${pf}_pa_${pa}_time_${time}/theta_${theta}_gamma_${gamma}/sitenum_${sitenum}_xi_${xi}/mix_in_${mix_in}_mass_matrix_theta_scale_${mass_matrix_theta_scale}_mass_matrix_gamma_scale_${mass_matrix_gamma_scale}_symplectic_integrator_num_steps_${symplectic_integrator_num_steps}_stepsize_${stepsize}/weight_${weight}_mass_matrix_weight_${mass_matrix_weight}/

                                                        if [ -f ./bash/${action_name}/pf_${pf}_pa_${pa}_time_${time}/theta_${theta}_gamma_${gamma}/sitenum_${sitenum}_xi_${xi}/mix_in_${mix_in}_mass_matrix_theta_scale_${mass_matrix_theta_scale}_mass_matrix_gamma_scale_${mass_matrix_gamma_scale}_symplectic_integrator_num_steps_${symplectic_integrator_num_steps}_stepsize_${stepsize}/weight_${weight}_mass_matrix_weight_${mass_matrix_weight}/run.sh ]; then
                                                            rm ./bash/${action_name}/pf_${pf}_pa_${pa}_time_${time}/theta_${theta}_gamma_${gamma}/sitenum_${sitenum}_xi_${xi}/mix_in_${mix_in}_mass_matrix_theta_scale_${mass_matrix_theta_scale}_mass_matrix_gamma_scale_${mass_matrix_gamma_scale}_symplectic_integrator_num_steps_${symplectic_integrator_num_steps}_stepsize_${stepsize}/weight_${weight}_mass_matrix_weight_${mass_matrix_weight}/run.sh
                                                        fi

                                                        mkdir -p ./bash/${action_name}/pf_${pf}_pa_${pa}_time_${time}/theta_${theta}_gamma_${gamma}/sitenum_${sitenum}_xi_${xi}/mix_in_${mix_in}_mass_matrix_theta_scale_${mass_matrix_theta_scale}_mass_matrix_gamma_scale_${mass_matrix_gamma_scale}_symplectic_integrator_num_steps_${symplectic_integrator_num_steps}_stepsize_${stepsize}/weight_${weight}_mass_matrix_weight_${mass_matrix_weight}/

                                                        touch ./bash/${action_name}/pf_${pf}_pa_${pa}_time_${time}/theta_${theta}_gamma_${gamma}/sitenum_${sitenum}_xi_${xi}/mix_in_${mix_in}_mass_matrix_theta_scale_${mass_matrix_theta_scale}_mass_matrix_gamma_scale_${mass_matrix_gamma_scale}_symplectic_integrator_num_steps_${symplectic_integrator_num_steps}_stepsize_${stepsize}/weight_${weight}_mass_matrix_weight_${mass_matrix_weight}/run.sh

                                                        tee -a ./bash/${action_name}/pf_${pf}_pa_${pa}_time_${time}/theta_${theta}_gamma_${gamma}/sitenum_${sitenum}_xi_${xi}/mix_in_${mix_in}_mass_matrix_theta_scale_${mass_matrix_theta_scale}_mass_matrix_gamma_scale_${mass_matrix_gamma_scale}_symplectic_integrator_num_steps_${symplectic_integrator_num_steps}_stepsize_${stepsize}/weight_${weight}_mass_matrix_weight_${mass_matrix_weight}/run.sh <<EOF
#!/bin/bash

#SBATCH --account=pi-lhansen
#SBATCH --job-name=${theta}_${gamma}_sn_${sitenum}_xi_${xi}_w_${weight}_mw_${mass_matrix_weight}_mix_in_${mix_in}_mix_${mix_in}_masst_${mass_matrix_theta_scale}_massg_${mass_matrix_gamma_scale}_step_${symplectic_integrator_num_steps}_${action_name}
#SBATCH --output=./job-outs/$job_name/${action_name}/pf_${pf}_pa_${pa}_time_${time}/theta_${theta}_gamma_${gamma}/sitenum_${sitenum}_xi_${xi}/mix_in_${mix_in}_mass_matrix_theta_scale_${mass_matrix_theta_scale}_mass_matrix_gamma_scale_${mass_matrix_gamma_scale}_symplectic_integrator_num_steps_${symplectic_integrator_num_steps}_stepsize_${stepsize}/weight_${weight}_mass_matrix_weight_${mass_matrix_weight}/run.out
#SBATCH --error=./job-outs/$job_name/${action_name}/pf_${pf}_pa_${pa}_time_${time}/theta_${theta}_gamma_${gamma}/sitenum_${sitenum}_xi_${xi}/mix_in_${mix_in}_mass_matrix_theta_scale_${mass_matrix_theta_scale}_mass_matrix_gamma_scale_${mass_matrix_gamma_scale}_symplectic_integrator_num_steps_${symplectic_integrator_num_steps}_stepsize_${stepsize}/weight_${weight}_mass_matrix_weight_${mass_matrix_weight}/run.err
#SBATCH --time=7-00:00:00
#SBATCH --partition=standard
#SBATCH --cpus-per-task=1
#SBATCH --mem=7G

module load python/booth/3.8

echo "\$SLURM_JOB_NAME"

echo "Program starts \$(date)"
start_time=\$(date +%s)


python3 -u /home/pengyu/HMC/HMC_leapfrog/$hmc_python_name  --pf ${pf} --pa ${pa} --time ${time} --theta ${theta} --gamma ${gamma} --sitenum ${sitenum} --xi ${xi} --weight ${weight} --dataname ${dataname} --mix_in ${mix_in} --mass_matrix_gamma_scale ${mass_matrix_gamma_scale} --mass_matrix_theta_scale ${mass_matrix_theta_scale}  --symplectic_integrator_num_steps ${symplectic_integrator_num_steps} --mass_matrix_weight ${mass_matrix_weight} --stepsize ${stepsize}
echo "Program ends \$(date)"
end_time=\$(date +%s)
elapsed=\$((end_time - start_time))

eval "echo Elapsed time: \$(date -ud "@\$elapsed" +'\$((%s/3600/24)) days %H hr %M min %S sec')"

EOF
                                                        count=$(($count + 1))
                                                        sbatch ./bash/${action_name}/pf_${pf}_pa_${pa}_time_${time}/theta_${theta}_gamma_${gamma}/sitenum_${sitenum}_xi_${xi}/mix_in_${mix_in}_mass_matrix_theta_scale_${mass_matrix_theta_scale}_mass_matrix_gamma_scale_${mass_matrix_gamma_scale}_symplectic_integrator_num_steps_${symplectic_integrator_num_steps}_stepsize_${stepsize}/weight_${weight}_mass_matrix_weight_${mass_matrix_weight}/run.sh
                                                    done
                                                done
                                            done
                                        done
                                    done
                                done
                            done
                        done
                    done
                done
            done
        done
    done
done
