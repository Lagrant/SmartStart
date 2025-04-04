# Smart Start

This project implements a hybrid initialization strategy for evolutionary algorithms (EAs) that combines Opposition-Based Learning (OBL) and the Empty-Space Search Algorithm (ESA) to address the critical impact of initialization on EA performance. OBL generates an initial diverse population, which is then refined by ESA to explore under-represented areas, resulting in enhanced population diversity, faster convergence, and improved solution quality, as demonstrated by benchmark results that outperform conventional initialization methods, particularly in complex, high-dimensional optimization problems.

## Requirements

All the requirments are specified in requirments.txt. 
Please refer to [COCO](https://github.com/numbbo/coco) to install and setup COCO benchmark.

## Run OBLESA

Run the following script in a bash:
```bash
python coco_benchmarks.py -o egwo -e 500 -s 1
```
Replace the arguments ```-o```, ```-e``` and ```-s``` with your own experiment settings. Explanations of the arguments can be found in coco_benchmarks.py

## Experiment results

Experiment results of the paper and parameters used in the algorithms are shown in [results](results.pdf).

## Authors

* [**Xinyu Zhang**](https://github.com/Lagrant/)

* [**Mário Antunes**](https://github.com/mariolpantunes)

* [**Tyler Estro**](https://www.fsl.cs.stonybrook.edu/~tyler/)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Copyright

This project is under the following [COPYRIGHT](COPYRIGHT).
