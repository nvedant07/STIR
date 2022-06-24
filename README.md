## Similarity Through Inverted Representations (STIR)  [Paper]()

**Measuring Representational Robustness of Neural Networks Through Shared Invariances**, ICML 2022.
[Vedant Nanda](https://nvedant07.github.io/), [Till Speicher](https://people.mpi-sws.org/~tspeicher/), [Camila Kolling](https://camilakolling.github.io/), [John P. Dickerson](http://jpdickerson.com/), [Krishna P. Gummadi](https://people.mpi-sws.org/~gummadi/), [Adrian Weller](http://mlg.eng.cam.ac.uk/adrian/)

### Installation

```
pip install stir-invariance
```

Or you can install it from source. First clone this repo and then run


```
(recommended) 
pip install -e .
```
or
```
python setup.py install
```

### Quick Start

```python
import stir
import stir.model.tools.helpers as helpers
import stir.helper as hp

test_dataloader = ... ## (instance of torch.utils.data.DataLoader, 
                      ## should return (images, labels) in each iter)
                      ## images should *not* be normalized since 
                      ## normalization is performed in model's forward pass

model1 = ... ## instance of torch.nn.Module
model1_dataset = 'cifar10'
normalizer1 = helpers.InputNormalize(*hp.DATASET_TO_MEAN_STD[model1_dataset]) 
## or any instance of torch.nn.Module that performs input normalization
model2 = ... ## instance of torch.nn.Module
model2_dataset = 'cifar10'
normalizer2 = helpers.InputNormalize(*hp.DATASET_TO_MEAN_STD[model2_dataset])
## or any instance of torch.nn.Module that performs input normalization

total_images = 1000 # number of images to use for computing STIR

# computes STIR between penultimate layer of model1 and model2
stir_score = stir.STIR(model1, model2, 
                       normalizer1, normalizer2, 
                       (test_dataloader, total_images))

stir_score.m1m2 ## STIR(m1|m2)
stir_score.m2m1 ## STIR(m2|m1)
stir_score.rsm ## Underlying similarity measure (default: Linear CKA)
```


## Citation
If you find our work useful, please cite it:

```
@inproceedings{nanda2022measuring,
    title={Measuring Representational Robustness of Neural Networks Through Shared Invariances},
    author={Nanda, Vedant and Speicher, Till and Kolling, Camilla and Dickerson, John P. and Gummadi, Krishna P. and Weller, Adrian},
    booktitle={ICML},
    year={2022}
}
```

## Acknowledgements
This repo borrows code from the [robustness library](https://github.com/MadryLab/robustness) to invert representations (and train models). The minibatch CKA implementation is inspired by [PyTorch-Model-Compare](https://github.com/AntixK/PyTorch-Model-Compare) and full batch from [@yuanli2333's CKA implementation](https://github.com/yuanli2333/CKA-Centered-Kernel-Alignment). Models for [TRADES](https://github.com/yaodongyu/TRADES) and [MART](https://github.com/YisenWang/MART) were trained using the repos made public by authors of the respective papers.
