# biomake

Create `biorbd`'s `bioMod`s of humans from a `yeadon` `meas.txt` measurements file.

## Installing

This script depends on the Python package `yeadon` for computing the Yeadon model of the human.
It also depends on `numpy` and `pyaml`.

They can be installed using `pip` or `conda`:
```
conda install -cconda-forge yeadon numpy pyyaml
```
or
```
python -m pip install yeadon numpy PyYAML
```

## Usage

Assuming you have your `Human.txt` for `yeadon`, you can simply
```
python biomake.py Human.txt > Human.bioMod
```
The `bioMod` created this way only contains the bare minimum: the mathematical skeleton of the Human.
To add degrees of freedom, meshes, markers, set default positions and other options of the `bioMod`,
you can specify a `Human_opt.yml` file to `biomake` this way
```
python biomake.py Human.txt --bioModOptions Human_opt.yml > Human.bioMod
```
Note however that `biomake` does not support `bioMod`'s "imu" nor "contact" declarations.

The created `bioMod`s can be visualized using `bioviz`.

See `example/` for sample `.yml` option files.

## References

For [`yeadon`](https://github.com/chrisdembia/yeadon)
> Dembia C, Moore JK and Hubbard M. An object oriented implementation of the Yeadon human inertia model, F1000Research 2014, 3:223 (doi: 10.12688/f1000research.5292.1)

For [`biorbd`](https://github.com/pyomeca/biorbd) and [`bioviz`](https://github.com/pyomeca/bioviz)
