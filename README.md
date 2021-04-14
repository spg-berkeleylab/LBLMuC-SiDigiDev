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
shifter --image=docker:infnpd/mucoll-ilc-framework:1.5-centos8 --volume=/global/cfs/cdirs/atlas/spgriso/MuonCollider/data:/data -- /bin/bash
```

#### Singularity
```bash
singularity shell --cleanenv /cvmfs/unpacked.cern.ch/registry.hub.docker.com/infnpd/mucoll-ilc-framework\:1.5-centos8
```

#### Docker (local)
```bash
MUC_CONTAINER_VER=1.5-centos8
PWD=path/to/working/directory
docker run -ti --rm --env DISPLAY=${DISPLAY} --env USER=${USER} --env HOME=/home/${USER} --env MUC_CONTAINER_VER=${MUC_CONTAINER_VER} --user=$(id -u $USER):$(id -g $USER) -v ${PWD}:/home/${USER} -v /cvmfs:/cvmfs -w /home/${USER} -v ${HOME}/.Xauthority:/home/${USER}/.Xauthority --net=host --entrypoint /bin/bash infnpd/mucoll-ilc-framework:${MUC_CONTAINER_VER}
```
where PWD is the path to your working directory. If this is your first time running this container, docker will pull it automatically, which takes a while.

Alternatively, use VSCode config files detailed in the VSCode setup section.

### Cloning repository
Just be careful that submodules are also cloned e.g.
```bash
git clone --recursive https://gitlab.cern.ch/berkeleylab/MuonCollider/LBLMuC-SiDigiDev.git
```

Note that the primary branch is called `main`.

Note that for submodule that are on `github`, soon only `SSH` keys will be allowed for authentication to push changes.
However, we need to keep the `HTTPS` for pulling the packages to make the pipeline to work without authentication; to solve this, once the packages are cloned, you can change only the `push` repository url as:
```bash
cd packages/LCTuple/
git remote set-url --push origin git@github.com:MuonColliderSoft/LCTuple.git
cd ../MuonCVXDDigitiser
git remote set-url --push origin git@github.com:MuonColliderSoft/MuonCVXDDigitiser.git
```
and so on for any other submodule that is hosted on `github`.

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
ddsim --steeringFile ${MYWORKSPACE}/config/sim_steer_muonGun_MuColl_v1.py
```

### Event Reconstruction
```bash
Marlin ${MYWORKSPACE}/config/reco_steer.xml
```

use instead `reco_steer_BIB.xml` for including BIB.

### Ntuple production
```bash
Marlin ${MYWORKSPACE}/config/lctuple_tracker.xml
```