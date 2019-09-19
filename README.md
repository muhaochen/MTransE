# Multilingual Knowledge Graph Embeddings for Cross-lingual Knowledge Alignment

This repository includes the code of MTransE var4 (see paper), links to the data sets, and pretrained models.

**A more recent tensorflow implementation** is available at this repository: https://github.com/muhaochen/MTransE-tf (**recommended**), which takes in entity-level seed alignment.
## Install
Make sure your local environment has the following installed:

    Python >= 2.7.6
    pip
    
Install the dependents using:

    ./install.sh

## Run the experiments
Please first download the data sets:

https://drive.google.com/open?id=1AsPPU4ka1Rc9u-XYMGWtvV65hF3egi0z

and pretrained models

https://drive.google.com/open?id=17JOLNlkkBqC5q14TwBBFLpflWusqJMak

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
        booktitle={Proceedings of the 26th International Joint Conference on Artificial Intelligence (IJCAI)},
        year={2017}
    }

## Links
The following links point to some recent follow-ups of this work.

Sun, Zequn, et al. [Cross-lingual entity alignment via joint attribute-preserving embedding.](https://iswc2017.semanticweb.org/wp-content/uploads/papers/MainProceedings/188.pdf) ISWC, 2017.  
Zhu, Hao, et al. [Iterative entity alignment via joint knowledge embeddings.](https://www.researchgate.net/profile/Hao_Zhu31/publication/318830326_Iterative_Entity_Alignment_via_Joint_Knowledge_Embeddings/links/598afe10aca27243585a115e/Iterative-Entity-Alignment-via-Joint-Knowledge-Embeddings.pdf), IJCAI, 2017.  
Yeo, Jinyoung, et al. [Machine-Translated Knowledge Transfer for Commonsense Causal Reasoning.](https://pdfs.semanticscholar.org/d065/0236b8cd7a693691eb479614d31a394b0c9b.pdf) AAAI. 2018.  
Chen, Muhao, et al. [Co-training Embeddings of Knowledge Graphs and Entity Descriptions for Cross-lingual Entity Alignment.](http://www.ijcai.org/proceedings/2018/0556.pdf), IJCAI, 2018.  
Sun, Zequn, et al. [Bootstrapping Entity Alignment with Knowledge Graph Embedding.](https://www.ijcai.org/proceedings/2018/0611.pdf) IJCAI. 2018.  
Otani, Naoki, et al. [Cross-lingual Knowledge Projection Using Machine Translation and Target-side Knowledge Base Completion.](http://www.aclweb.org/anthology/C18-1128) COLING, 2018.  
Wang, Zhichun, et al. [Cross-lingual Knowledge Graph Alignment via Graph Convolutional Networks.](http://aclweb.org/anthology/D18-1032) EMNLP, 2018.  
Trsedya, Bayu D, et al. [Entity Alignment between Knowledge Graphs Using Attribute Embeddings.](http://www.ruizhang.info/publications/AAAI2019-Entity%20Alignment%20between%20Knowledge%20Graphs%20Using%20Attribute%20Embeddings.pdf) AAAI, 2019.  
Qu, M., Tang, J., Bengio, Y. [Weakly-supervised Knowledge Graph Alignment with Adversarial Learning](https://arxiv.org/abs/1907.03179).  
Hao, J., et al. [Universal Representation Learning of Knowledge Bases by Jointly Embedding Instances and Ontological Concepts.](http://yellowstone.cs.ucla.edu/~muhao/articles/KDD19_JOIE.pdf) KDD, 2019.  
Guo, L., et al. [Learning to Exploit Long-term Relational Dependencies in Knowledge Graphs](https://arxiv.org/pdf/1905.04914.pdf). ICML, 2019.  
Zhang, Q., et al. [Multi-view Knowledge Graph Embedding for Entity Alignment.](https://arxiv.org/pdf/1906.02390.pdf) IJCAI, 2019.  
Zhu, Q., et al. [Neighborhood-Aware Attentional Representation for Multilingual Knowledge Graphs](https://www.ijcai.org/proceedings/2019/0269.pdf) IJCAI, 2019.  
Pei, S., et al. [Improving Cross-lingual Entity Alignment via Optimal Transport](https://www.ijcai.org/proceedings/2019/0448.pdf) IJCAI, 2019.  
Wu, Y., et al. [Relation-Aware Entity Alignment for Heterogeneous Knowledge Graphs](https://www.ijcai.org/proceedings/2019/0733.pdf) IJCAI, 2019.  
Pei, C., et al. Semi-Supervised Entity Alignment via Knowledge Graph Embedding with Awareness of Degree Difference. WWW, 2019.  
Xu, Kun, et al. [Cross-lingual Knowledge Graph Alignment via Graph Matching Neural Network](https://arxiv.org/pdf/1905.11605). ACL, 2019.  
Cao, Yi., et al. [Multi-Channel Graph Neural Network for Entity Alignment.](https://www.aclweb.org/anthology/P19-1140) ACL, 2019.  
Sun, Z., et al. TransEdge: Translating Relation-contextualized Embeddings for Knowledge Graphs. ISWC, 2019.  
Yang, H., et al. [Aligning Cross-Lingual Entities with Multi-Aspect Information.](https://cs.uwaterloo.ca/~jimmylin/publications/YangHW_etal_EMNLP2019.pdf) EMNLP, 2019
