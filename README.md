
## Code 

The optimized code can be found in the `opti` directory. The openmp `parMDS.cpp` code is optimized for better performance. The cuda `parMDS.cu` code is just direct port of the original `parMDS.cpp` code from the root directory.

## Docs

The latex report and major project presentation can be found in `/docs` folder

## Results 

The `results` dir contains the outputs of the runs done on original parMDS.cpp (`openmp`), optimized opti/parMDS.cpp (`mst`) and cuda opti/parMDS.cu (`cuda`). Comparative analysis can be found in `results/plots`

> Ignore : The `vrp` folder is just an attempt to visualize the customers and depot to intuitively understand the problem and how the routes are defined. The `.html` file will be an intereactive file to load (hopefully) thousands of nodes and can be interacted visually. The source for this is at `viz/vrp.py`

### Build and run all rounds.

```shell
bash ./runRounds.sh # this will run the original openmp, opti cuda and opti openmp code and compare them on all the vrp instances in `/inputs`
```