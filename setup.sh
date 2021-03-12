if [ ${#} != 0 ] && [ ${#} != 1 ]; then
    echo "usage: ${0} [build]"
    return 1
fi

#
# Check for build directory
MYBUILD=build
if [ ${#} == 1 ]; then
    MYBUILD=${1}
fi

if [ ! -d ${MYBUILD} ]; then
    echo "Build directory ${MYBUILD} does not exist!"
    return 1
fi

# Convert to absolute path
export MYWORKSPACE=$(realpath $(dirname ${BASH_SOURCE[0]}))
export MYBUILD=$(realpath ${MYBUILD})

#
# Main software
source /opt/ilcsoft/muonc/init_ilcsoft.sh

#
# Add exts
export LD_LIBRARY_PATH="$(find ${MYBUILD}/exts/* -name lib64 -type d | tr '\n' ':')$(find ${MYBUILD}/exts/* -name lib -type d | tr '\n' ':')${LD_LIBRARY_PATH}"
export PATH="$(find ${MYBUILD}/exts/*/bin -type d | tr '\n' ':')${PATH}"

# Modify MARLIN_DLL for packages
for pkg in `ls -1 -d packages/* | tr '\n' '\0' | xargs -0 -n 1 basename`; do
    # check if package is in MARLIN_DLL
    FOUND=0
    echo ${MARLIN_DLL} | grep '\(:\|^\)\([^:]*\/'$pkg'\/[^:]*\)\(:\|$\)' && FOUND=1
    if [ ${FOUND} -eq 1 ]; then
	#Package inside MARLIN_DLL, replace string
	export MARLIN_DLL=`echo $MARLIN_DLL | sed 's/\(:\|^\)\([^:]*\/'$pkg'\/[^:]*\)\(:\|$\)/\1'${MYBUILD}/packages/$pkg/lib/lib${pkg}.so'\3/'`
    else
	#Package not in MARLIN_DLL, add it
	export MARLIN_DLL="${MYBUILD}/packages/$pkg/lib/lib${pkg}.so:${MARLIN_DLL}"
    fi
done
