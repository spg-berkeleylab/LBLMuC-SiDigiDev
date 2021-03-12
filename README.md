# LBL Muon Collider Workspace

Collection of packages for Muon Collider studies done at LBL for silicon digitization developments.

## Repository Structure
- `exts/` External packages not included with the ILC framework.
- `packages/` All non-standard packages linked using git submodules.
- `sim/` Configuration files for generating events.
- `analysis/` Performance analysis macros

## Setup Instructions

### Container
All commands should be run inside the `infnpd/mucoll-ilc-framework:1.5-centos8` container.

#### Shifter (cori)
```bash
shifter --image=docker:infnpd/mucoll-ilc-framework:1.5-centos8 --volume=/global/cfs/cdirs/atlas/spgriso/MuonCollider/tutorial-data:/data -- /bin/bash
```

#### Singularity
```bash
singularity shell --cleanenv /cvmfs/unpacked.cern.ch/registry.hub.docker.com/infnpd/mucoll-ilc-framework\:1.5-centos8
```

### Build Instructions
Run the following commands from inside the container. The same commands will also work with a local installation of the ILC software, with the exception of the first line.
```bash
source /opt/ilcsoft/muonc/init_ilcsoft.sh # Setup ILC software
cmake -S . -B build -DBOOST_INCLUDEDIR=/usr/include/boost169 -DBOOST_LIBRARYDIR=/usr/lib64/boost169 
cmake --build build
```

### Setup Script
The included `setup.sh` script is useful for defining all paths for the binaries built by the workspace. At the current stage, it setups the following:
- ILC software via `init_ilcsoft.sh`
- External binaries/libraries found in `exts`.
- Add all package libraries to `MARLIN_DLL`.

Run the following at the start of every session. It has an optional argument to the build directory and is set to `build/` by default.
```bash
source setup.sh [build]
```

## Useful Commands

Run the following commands from your run directory. Make sure to replace `${MYBUILD}` and `${MYWORKSPACE}` with the absolute paths to your build and workspace directories.

### Event Simulation
```bash
ddsim --steeringFile ${MYBUILD}/sim/sim_steer_muonGun_MuColl_v1_mod0.py
```

