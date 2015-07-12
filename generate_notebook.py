
import IPython.nbformat.current as nbf

py_script_name = 'analysis'
nb = nbf.read(open(py_script_name + '.py', 'r'), 'py')
nbf.write(nb, open(py_script_name + '.ipynb', 'w'), 'ipynb')
print("Wrote a python notebook {0}, from script: {1}.".format(py_script_name + '.ipynb', py_script_name + '.py'))
