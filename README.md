# LBL Muon Collider Workspace

Collection of packages for Muon Collider studies done at LBL.

## Repository Structure
- `exts/` External packages not included with the ILC framework.
- `packages/` All non-standard packages linked using git submodules.
- `sim/` Configuration files for generating events.

## Setup Instructions

### Container
All commands should be run inside the `infnpd/mucoll-ilc-framework:1.4-centos8` container.

#### Singularity
```bash
singularity shell --cleanenv /cvmfs/unpacked.cern.ch/registry.hub.docker.com/infnpd/mucoll-ilc-framework\:1.0-centos8
```

### Build Instructions
Run the following commands from inside your `mucoll-ilc-framework:1.4-centos8` container. The same commands will also work with a local installation of the ILC software, with the exception of the first line.
```bash
source /opt/ilcsoft/muonc/init_ilcsoft.sh # Setup ILC software
cmake -S . -B build
cmake --build build
```

### Setup Script
The included `setup.sh` script is useful for defining all paths for the binaries built by the workspace. At the current stage, it setups the following:
- ILC software via `init_ilcsoft.sh`
- External binaries/libraries found in `exts`.

Run the following at the start of every session. It has an optional argument to the build directory and is set to `build/` by default.
```bash
source setup.sh [build]
```

## Example Commands

Run the following commands from your run directory. Make sure to replace `${MYBUILD}` and `${MYWORKSPACE}` with the absolute paths to your build and workspace directories.

### Exporting Geometry to TGeo
```bash
${MYBUILD}/packages/ACTSMCC/dd2tgeo ${MYWORKSPACE}/data/MuColl_v1_mod0/MuColl_v1_mod0.xml MuColl_v1_mod0.root
```

### Event Simulation
```bash
ddsim --steeringFile ${MYBUILD}/data/sim_steer_muonGun_MuColl_v1_mod0.py
```

### Dump Events Into CSV Files
```bash
MARLIN_DLL=${MYBUILD}/packages/ACTSTuple/libACTSTuple.so:${MARLIN_DLL} Marlin ${MYBUILD}/packages/ACTSTuple/example/actstuple_steer.xml --global.LCIOInputFiles=/path/to/events.slcio
```

## Modified Geometry Destription
The workspace contains a modified geometry description that makes it usable with the ACTS tracking framework. The modification are to logical description only and do not affect the physical properties of the detector.

### lcgeo Modifications
The sensitive parts of the detector have been renamed from `component#` to `sensor#`. This makes it easier to identify the sensor inside the TGeo geometry used by ACTS.

The modified `lcgeo` package is stored under `exts/`.

### XML Modifications
The inner tracker needs to be split into two parts to prevent overlap in the cylindrical bounding volumes. The modified geometry, `data/MuColl_v1_mod0`, adds a `OuterInnerTracker` subdetector with the outer layers in the inner tracker.
