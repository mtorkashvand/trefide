#!/usr/bin/env bash
set -euo pipefail
# export DOCKER_HOST=tcp://localhost:2375

echo Test that Docker is reachable:
docker ps

# https://stackoverflow.com/questions/360201/how-do-i-kill-background-processes-jobs-when-my-shell-script-exits

# If we stop with <C-c> then we want to also clean up/stop any docker processes
trap 'exit' INT TERM

# upon EXIT signal, kill all spawned processes and also `docker kill pipeline` if necessary
trap on_exit EXIT

on_exit() {
  spawned_jobs=$(jobs -p)
  [ -n "$spawned_jobs" ] && kill $spawned_jobs
  if [ "$(docker inspect -f '{{.State.Running}}' pipeline)" == "true" ]; then
    docker kill pipeline || echo 'pipeline' container not found
  fi
}

############################################################
# Function definitions
############################################################

usage(){

  echo "Usage: bash $0 [-a] [-b] [-s] [-t] [-f] [-o] [-k] <-i 'infile'> <-d 'input dir'> <-n 'name_for_run'> <-m 'image name'> [-x 'cut_off_point values'] [-y 'length_cut values'] [-z 'corr_th_fix values'] [-h] 

  -a : means do denoising
  -b : means do demixing
  -d : name of input dir
  -i : name of input file
  -f : set flag to take input from MATLAB script (incompatible with '-s' flag)
  -o : set flag to use HDF5 format for final output (for MATLAB script)
  -k : set flag to skip preprocessing (for MATLAB script)
  -s : save diagnostic figures
  -t : set tiny-mode for testing. NOT CURRENTLY WORKING!
  -n : name for results folder
  -m : name of docker image to use
  -x, -y, -z : pass parameters for grid searching the demix parameters
  -h : print usage

  EXAMPLES:

  Run everything, saving diagnostics:
  'bash $0 -ab -s -i infile.mat -d /d/Documents -n test_run -m paninski-segmentation'

  Run as part of MATLAB script:
  (not saving diagnostics, input from matlab and output hdf5, skip preprocess)
  'bash $0 -ab -f -o -k -i infile.mat -d /path/to/tmp -n tmp -m paninski-segmentation'

  Run denoising once, then troubleshoot demixing (overwriting previous demixing results):
  'bash $0 -a -s -i infile.mat -d /d/Documents -n editing_demixing -m paninski-segmentation'
  'bash $0 -b -s -i infile.mat -d /d/Documents -n editing_demixing -m paninski-segmentation'
  ...
  <make changes>
  ...
  'bash $0 -b -s -i infile.mat -d /d/Documents -n editing_demixing -m paninski-segmentation'
  "
  # TODO - This grid search scenario was previously possible with '-q' flag, but might need adjustments to work right now.
  # For example, could add a folder suffix option and put the demix results in a different subfolder

  #Run the first steps once, then grid search params on the second steps
  #'bash $0 -a -s -i infile.mat -n demix_grid_search'
  #'for x in 1 2 3 4; do bash $0 -b -s -i chopped_fullsize.mat -n demix_grid_search ; done'
}

start_docker_stats() {
  # Example one-liner to browse `docker stats` output:
  # cat /d/Documents/Data/niklas/UCB-105_Sentinel/dev-tmp/results.2019-07-15.14-42-24/docker_stats.5s.csv | cut -d',' -f 2 | cut -d'=' -f 2 | cut -d'/' -f 1 | grep -v 'MiB' | tail -n+2 | cut -d'G' -f1 | sort -k1,1n | tail -n10
  # grab the PID with `docker_stats_pid=$!` after using this function
  [[ $1 ]] || { echo "Missing filename for start_docker_stats()!" >&2 ; exit 1 ; }

  # CSV header
  echo "CPUPerc,MemUsage,BlockIO,PIDs" > ${1}

  # fetch stats every 5s
  nohup bash -c "while true; do docker stats --no-stream --format \"{{.CPUPerc}},{{.MemUsage}},{{.BlockIO}},{{.PIDs}}\" >> ${1}; sleep 5; done" > /dev/null 2>&1 &
}

docker_wait(){
  set +e
  docker inspect pipeline >/dev/null 2>&1
  exists_zero=$?  # exit code '0' if container exists
  set -e

  # we run the container detached so we cannot accidentally kill it
  # however we want to fail the pipeline if it fails
  exit_code=0
  if [[ $exists_zero -eq 0 ]]; then
    docker wait pipeline # works even if container already stopped
    exit_code=$(docker inspect pipeline --format='{{.State.ExitCode}}')
    docker rm pipeline
  fi
  return $exit_code  # TODO - if we are beginning from downstream stage, and previously the container failed, then we will immediately fail.
                     # Instead, the first "docker_wait" of any pipeline run should succeed no matter what the previous container exitcode was
}

run_denoise_analysis(){
  #if [[ $1 -eq 0 ]]; then # we were instructed that previous errors do not matter
    # docker_wait || echo 'previous container exited with error, but we do not care here'
  #  echo 'previous container exited with error, but we do not care here'
  #else
  #  docker_wait
  #fi

  docker_stats_file="${outdir}/docker_stats.denoise.5s.csv"
  start_docker_stats ${docker_stats_file}
  docker_stats_pid=$!

  # docker run --name pipeline -d \
  #            -u $(id -u ${USER}):$(id -g ${USER}) \
  #            -v ${indir}/${infile}:/input/${infile}:ro \
  #            -v ${outdir}/:/output/ ${image_name} bash \
  #            "source activate root && \
  #            python ~/trefide/qfunctions/run_denoise.py \
  #            --infile ${infile} \
  #            ${tiny_mode} \
  #            ${input_matlab} \
  #            ${save_diagnostics} \
  #            ${skip_preprocess} \
  #            --median-window 21 \
  #            --consec-failures 3 \
  #            --max-iters-main 10 \
  #            --max-iters-init 40 \
  #            --d-sub 2 \
  #            --t-sub 2 \
  #            --block-width 40 \
  #            --block-height 40 \
  #            --max-components 50"


  docker run --name pipeline -d \
  -v ${indir}/${infile}:/input/${infile}:ro \
  -v ${outdir}/:/output/ \
  ${image_name}

  # added to update use the most recent version
  # docker exec pipeline /bin/bash -c \
  # "cd ~/trefide ; git pull"
  # docker exec pipeline /bin/bash -c \
  # "source activate root && \
  # python /trefide_repo/qfunctions/run_denoise.py \
  # --infile ${infile} \
  # ${tiny_mode} \
  # ${input_matlab} \
  # ${save_diagnostics} \
  # ${skip_preprocess} \
  # --median-window 21 \
  # --consec-failures 3 \
  # --max-iters-main 10 \
  # --max-iters-init 40 \
  # --d-sub 2 \
  # --t-sub 2 \
  # --block-width 40 \
  # --block-height 40 \
  # --max-components 50"
            
  docker exec pipeline /bin/bash -c \
  "source activate root && \
  python ~/trefide/qfunctions/run_denoise.py \
  --infile ${infile} \
  ${tiny_mode} \
  ${input_matlab} \
  ${save_diagnostics} \
  ${skip_preprocess} \
  --median-window 21 \
  --consec-failures 3 \
  --max-iters-main 10 \
  --max-iters-init 40 \
  --d-sub 2 \
  --t-sub 2 \
  --block-width 40 \
  --block-height 40 \
  --max-components 50"

  docker wait pipeline && kill -9 ${docker_stats_pid} &
  docker logs -f pipeline
}

run_demix_analysis(){
  # if [[ $1 -eq 0 ]]; then # we were instructed that previous errors do not matter
    # docker_wait || echo 'previous container exited with error, but we do not care here'
  #   echo 'previous container exited with error, but we do not care here'
  # else
  #   docker_wait
  # fi

  docker_stats_file="${outdir}/docker_stats.demix.5s.csv"
  start_docker_stats ${docker_stats_file}
  docker_stats_pid=$!

  docker run --name pipeline -d \
  -v ${indir}/${infile}:/input/${infile}:ro \
  -v ${outdir}/:/output/ \
  ${image_name}

  docker exec pipeline /bin/bash -c \
  "source activate root && \
  python ~/trefide/qfunctions/run_demix.py \
  --infile ${infile} \
  ${tiny_mode} \
  ${save_diagnostics} \
  ${input_matlab} \
  ${output_hdf5} \
  --cut-off-point ${cut_off_point} \
  --length-cut ${length_cut} \
  --th 2 1 \
  --pass-num 2 \
  --residual-cut 0.6 0.6 \
  --corr-th-fix ${corr_th_fix} \
  --max-allow-neuron-size 0.3 \
  --merge-corr-thr 0.6 \
  --merge-overlap-thr 0.6 \
  --num-plane 1 \
  --patch-size 100 100 \
  --plot-en false \
  --TF false \
  --fudge-factor 1 \
  --text true \
  --bg false \
  --max-iter 35 \
  --max-iter-fin 50 \
  --update-after 4"

  # docker run --name pipeline -d \
  # -u $(id -u ${USER}):$(id -g ${USER}) \
  # -v ${indir}/${infile}:/input/${infile}:ro \
  # -v ${outdir}/:/output/ \
  # ${image_name} bash -c "source activate root && \
  # python ~/trefide/qfunctions/run_demix.py \
  # --infile ${infile} \
  # ${tiny_mode} \
  # ${save_diagnostics} \
  # ${input_matlab} \
  # ${output_hdf5} \
  # --cut-off-point ${cut_off_point} \
  # --length-cut ${length_cut} \
  # --th 2 1 \
  # --pass-num 2 \
  # --residual-cut 0.6 0.6 \
  # --corr-th-fix ${corr_th_fix} \
  # --max-allow-neuron-size 0.3 \
  # --merge-corr-thr 0.6 \
  # --merge-overlap-thr 0.6 \
  # --num-plane 1 \
  # --patch-size 100 100 \
  # --plot-en false \
  # --TF false \
  # --fudge-factor 1 \
  # --text true \
  # --bg false \
  # --max-iter 35 \
  # --max-iter-fin 50 \
  # --update-after 4"


  docker wait pipeline && kill -9 ${docker_stats_pid} &
  docker logs -f pipeline
}

do_denoise_analysis=0
do_demix_analysis=0

echo $#

if [[ $# -eq 0 ]]; then
  echo "Choose at least 1 stage to run"
fi

############################################################
# Run parameters
############################################################
TIME_STRING=$(date +"%Y-%m-%d.%H-%M-%S")
#indir=/d/Documents/data/niklas/UCB-105_Sentinel/dev-tmp/qsm
# TODO - hardcoded base directory for input files
#indir=/home/niklas/git/funimag

cut_off_point="0.95 0.9" # need quotes
length_cut="15 10" # need quotes
corr_th_fix=0.31
#q_flag=0
#o_flag=0
tiny_mode=' '
save_diagnostics=' '
output_hdf5=' '
input_matlab=' '
skip_preprocess=' '

while getopts "abhstofki:d:n:m:x:y:z:" optchar; do
    case "${optchar}" in
        h) usage ;;
        a) do_denoise_analysis=1 ;;
        b) do_demix_analysis=1 ;;
        i) infile=${OPTARG} ;;
        d) indir=${OPTARG} ;;
        o) output_hdf5='--output-hdf5' ;;
        f) input_matlab='--input-matlab' ;;
        k) skip_preprocess='--skip-preprocess' ;;
        n) name=${OPTARG} ;;
        m) image_name=${OPTARG} ;;
        s) save_diagnostics='--save-diagnostics' ;;

        # TODO - tiny mode probably broken
        t) tiny_mode='--tiny-mode' 
           echo "TINY MODE NOT CURRENTLY WORKING!" >&2;
           exit 1;;

        # x, y, z flags for grid searching the key demix parameters
        x) cut_off_point="${OPTARG}" ;;
        y) length_cut="${OPTARG}" ;;
        z) corr_th_fix="${OPTARG}" ;;

#        o) # re-use the same outdir and replace the contents as needed
#           o_flag=1
#           o_optarg=${OPTARG} ;;
#        q) # copy the contents to a new folder and continue there
#           q_flag=1 
#           q_optarg=${OPTARG} ;;
        *) echo "Bad options" >&2; usage ; exit 1;;
    esac
done

#if [[ $o_flag -eq 1 ]] && [[ $q_flag -eq 1 ]]; then
#  echo "Choose either -o OR -q, not both" >&2
#  usage
#  exit 1
#fi

# Sanity check the provided dir+file
# ls ${indir}/${infile}

outdir="${indir}/${name}"
echo "OUTPUTS WILL BE PLACED AT ${outdir}"
#if [[ $o_flag -eq 1 ]]; then
#  outdir=${indir}/${o_optarg}
#fi

mkdir -p ${outdir}
# replacing spaces with underscores using bash built-in syntax: ${variable// /_}
touch "${outdir}/cut_off_point_${cut_off_point// /_}.length_cut_${length_cut// /_}.corr_th_fix_${corr_th_fix// /_}.FLAG"

#if [[ $q_flag -eq 1 ]]; then
#  original_dir=${indir}/${q_optarg} 
#  cp -R ${original_dir}/* ${outdir}
#fi

# TODO - don't need to mount the same volume twice - only need 2 volume mounts if separate loactions
# TODO - can docker volume/file mount an s3fs driver or another remote file source?

# Write git status
#echo git commit: $(git rev-parse HEAD) > ${outdir}/git_status.txt 2>&1
#local_changes=$(git status --porcelain)
# { 
#if [[ -z ${local_changes} ]]; then
#  echo Clean
#else
#  echo ${local_changes}
#fi ; } >> ${outdir}/git_status.txt 2>&1

############################################################
# Pipeline control
############################################################

# This flag variable controls how we should handle the previous exit code of the `pipeline` container.
# '0' means ignore previous errors
# '1' means we should crash if we see that the container exited with error (e.g. during middle of a pipeline run!)
ignore_previous_errors=0 

if [[ $do_denoise_analysis -eq 1 ]]; then
  echo do_denoise_analysis
  run_denoise_analysis ignore_previous_errors
  # ignore_previous_errors=1
fi

if [[ $do_demix_analysis -eq 1 ]]; then
  echo do_demix_analysis
  run_demix_analysis ignore_previous_errors
fi
