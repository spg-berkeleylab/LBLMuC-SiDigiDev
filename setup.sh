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

#
# Add new modules
for pkglib in $(find ${MYBUILD}/packages -name '*.so' -type l -o -name '*.so' -type f)
do
    pkgname=$(basename ${pkglib})    
    # check if package is in MARLIN_DLL
    FOUND=0
    echo ${MARLIN_DLL} | grep '\(:\|^\)\([^:]*\/'$pkgname'\/[^:]*\)\(:\|$\)' && FOUND=1
    if [ ${FOUND} -eq 1 ]; then
	#Package inside MARLIN_DLL, replace string
	MARLIN_DLL=`echo $MARLIN_DLL | sed 's/\(:\|^\)\([^:]*\/'$pkg'\/[^:]*\)\(:\|$\)/\1'${pkglib}'\3/'`
    else
	#Package not in MARLIN_DLL, add it
	MARLIN_DLL="${MYBUILD}/packages/$pkg/lib/lib${pkg}.so:${MARLIN_DLL}"
    fi
done
export MARLIN_DLL
