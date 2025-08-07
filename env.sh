centOSversion=$(awk -F= '/VERSION_ID/ {split($2, version, "."); print version[1]}' /etc/os-release | tr -d '"')
if [ "$centOSversion" -eq 7 ]; then
	source /cvmfs/sft.cern.ch/lcg/views/LCG_102b/x86_64-centos7-gcc11-opt/setup.sh
else
  source /cvmfs/sft.cern.ch/lcg/views/LCG_105b/x86_64-el9-gcc12-opt/setup.sh
fi

export DELPHES_PATH=/eos/home-h/hyeh/MG5_aMC_v3_5_4/Delphes/
export LD_LIBRARY_PATH=$DELPHES_PATH/lib:$LD_LIBRARY_PATH
export PATH=$DELPHES_PATH/bin:$PATH

######
#source /cvmfs/sft.cern.ch/lcg/views/LCG_102b/x86_64-centos9-gcc11-opt/setup.sh
#source /cvmfs/sft.cern.ch/lcg/views/LCG_103/x86_64-centos7-gcc11-opt/setup.sh
######