# geometrically-reduced-PVS-flow

Simulation code, meshes and associated data to reproduce numerical examples presented in 
"Geometrically reduced modelling of pulsatile flow in
perivascular networks", C. Daversin-Catty, I. G. Gjerde and M.E. Rognes (2021)

## Running models

The 1D and 3D simulations are performed using CFD models based on the [FEniCS project](https://fenicsproject.org/).
The model used for the 3D simulations is the one used in ["The mechanisms behind perivascular fluid flow", C. Daversin-Catty, V. Vinje, K-A. Mardal, and M.E. Rognes (2020)](https://journals.plos.org/plosone/article/comments?id=10.1371/journal.pone.0244442).

The corresponding Python scripts can be run within the latest [FEniCS Docker container](https://quay.io/repository/fenicsproject/dev)
with the last version of [Scipy](https://www.scipy.org/) installed :
```
git clone https://github.com/cdaversin/geometrically-reduced-PVS-flow
docker run -it -v $(pwd)/geometrically-reduced-PVS-flow:/home/fenics/shared quay.io/fenicsproject/dev
sudo pip3 install scipy --upgrade
cd shared
```

### Domain A
The A1 and A2 model can be configured and run using the corresponding scripts
```
cd src/2D_to_1D/scripts
python3 script_A1.py
python3 script_A2.py
```

### Domain B and C
The B2, B2, B3 and C12 models can be configured and run using the corresponding scripts
```
cd src/3D_to_1D/scripts
python3 script_B1.py
python3 script_B2.py
python3 script_B3.py
python3 script_C12.py
```

## Graphs
The graphs presented in the paper can be reproduced using [Jupyter notebook](https://jupyter.org/),
running the corresponding scripts in a Web browser.
```
cd geometrically-reduced-PVS-flow/notebooks
jupyter-notebook
```
Note : The data files used in the notebooks are present in the repository by default, and are re-generated
when running the models as described in the [dedicated section](#models)

## Mesh generation
The generation of the 3D PVS meshes presented in the paper is performed using [PVS-meshing-tools](https://github.com/cdaversin/PVS-meshing-tools), based on [VMTK](http://www.vmtk.org/)
on clipped geometries from [Aneurisk dataset](http://ecm2.mathcs.emory.edu/aneuriskweb/repository) (cases id C0092 and C0075).

## Reporting issues
Any questions regarding this repository can be posted as [Issues](https://github.com/cdaversin/geometrically-reduced-PVS-flow/issues).
