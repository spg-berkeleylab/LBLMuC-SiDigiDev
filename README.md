# LBL Muon Collider Workspace

Collection of packages for Muon Collider studies done at LBL.

## Repository Structure
- `packages/` All non-standard packages linked using git submodules.
- `sim/` Configuration files for generating events.

## Setup Instructions

### Container
All commands should be run inside the `mucoll-ilc-framework:1.0-centos8` container.

#### Singularity
```bash
singularity shell --cleanenv /cvmfs/unpacked.cern.ch/registr  
y.hub.docker.com/infnpd/mucoll-ilc-framework\:1.0-centos8
```

### Build Instructions
Run the following commands from inside your `mucoll-ilc-framework:1.0-centos8` container. The same commands will also work with a local installation of the ILC software, with the exception of the first line.
```bash
source /opt/ilcsoft/v02-01-pre/init_ilcsoft.sh # Setup ILC software
mkdir build
cmake ..
make
```

## Example Commands

Run the following commands from your run directory. Make sure to replace `${MYBUILD}` and `${MYWORKSPACE}` with the absolute paths to your build and workspace directories.

### Dump Events Into CSV Files
```bash
MARLIN_DLL=${MYBUILD}/packages/ACTSTuple/libACTSTuple.so:${MARLIN_DLL} Marlin ${MYWORKSPACE}/pa  
ckages/ACTSTuple/example/actstuple_steer.xml --global.LCIOInputFiles=/path/to/events.slcio
```