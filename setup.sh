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
export MYBUILD=$(realpath ${MYWORKSPACE}/${MYBUILD})

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
    if [[ "${MARLIN_DLL}" == *"${pkgname}"* ]]; then
        echo "Replacing existing $pkgname in MARLIN_DLL"
        MARLIN_DLL=$(echo ${MARLIN_DLL} | sed -e 's|[^:]\+'${pkgname}'|'${pkglib}'|')
    else
        echo "Adding $pkgname to MARLIN_DLL"
        MARLIN_DLL=${pkglib}:${MARLIN_DLL}
    fi

    # # check if package is in MARLIN_DLL
    # FOUND=0
    # echo ${MARLIN_DLL} | egrep '(:|^)[^:]*/'$pkgname'[^:]*(:|$)' > /dev/null && FOUND=1
    # if [ ${FOUND} -eq 1 ]; then
    #     echo "Replacing existing $pkgname in MARLIN_DLL"
	#     #Package inside MARLIN_DLL, replace string
	#     MARLIN_DLL=`echo $MARLIN_DLL | sed 's/\(:\|^\)\([^:]*\/'$pkgname'\/[^:]*\)\(:\|$\)/\1'${pkglib}'\3/'`
    # else
    #     echo "Adding $pkgname to MARLIN_DLL"
	#     #Package not in MARLIN_DLL, add it
	#     MARLIN_DLL="${pkglib}:${MARLIN_DLL}"
    # fi
done
export MARLIN_DLL
