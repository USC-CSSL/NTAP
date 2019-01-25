
VENV=venv
PYPATH=$(command -v python3)
USE_GPU=false

if [ ! -d $VENV ]
then
    virtualenv $VENV -p $PYPATH
    source ${VENV}/bin/activate
    pip install Cython
    if [ $USE_GPU = true ] ; then
        pip install -r requirements_gpu.txt
    else
        pip install -r requirements.txt
    fi
else
    echo "Environment already created at ${VENV}"
    source ${VENV}/bin/activate
fi

