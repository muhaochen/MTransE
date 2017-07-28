# Multilingual Knowledge Graph Embeddings for Cross-lingual Knowledge Alignment

This repository includes the code of MTransE var4 (see paper), links to the data sets, and pretrained models.
## Install
Make sure your local environment has the following installed:

    Python >= 2.7.6
    pip
    
Install the dependents using:

    ./install.sh

## Run the experiments
Please first download the data sets:
http://yellowstone.cs.ucla.edu/~muhao/MTransE/data.zip
and pretrained models
http://yellowstone.cs.ucla.edu/~muhao/MTransE/models.zip
Unpack these two folders to the local clone of the repository.
To run the experiments on WK3l (wikipedia graphs), use:

    ./run_wk3l.sh
To run the experiments on CN3l (conceptNet), use:

    ./run_cn3l.sh
You may also train your own models on these two data sets using:

    ./train_models.sh

## Reference
Please refer to our paper. 
Muhao Chen, Yingtao Tian, Mohan Yang, Carlo Zaniolo. Multilingual Knowledge Graph Embeddings for Cross-lingual Knowledge Alignment. In *Proceedings of the 26th International Joint Conference on Artificial Intelligence (IJCAI)*, 2017

    @inproceedings{chen2017multigraph,
        title={Multilingual Knowledge Graph Embeddings for Cross-lingual Knowledge Alignment},
        author={Chen, Muhao and Tian, Yingtao and Yang, Mohan and Zaniolo, Carlo},
        booktitle={Fuzzy Systems (FUZZ-IEEE), 2016 IEEE International Conference on},
        year={2016}
    }
   
    