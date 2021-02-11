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
MYBUILD=$(realpath ${MYBUILD})

#
# Main software
source /opt/ilcsoft/muonc/init_ilcsoft.sh

#
# Add exts
LD_LIBRARY_PATH="$(find ${MYBUILD}/exts/*/lib -type d | tr '\n' ':')${LD_LIBRARY_PATH}"
PATH="$(find ${MYBUILD}/exts/*/bin -type d | tr '\n' ':')${PATH}"
