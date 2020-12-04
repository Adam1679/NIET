# NIET: Exploring Neighborhood Information for Knowledge Graph Entity Typing

### Requirements
- [conda](https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh)

Please download miniconda from above link and create an environment using the following command:

        conda env create -f pytorch35.yml

Activate the environment before executing the program as follows:

        source activate pytorch35

### Dataset
We used two different datasets for evaluating our model. All the datasets and their folder names are given below.
- FB15k
- YAGO43k

### Training

**Parameters:**
`--batch_size_gat`: Batch size for GAT.
`--valid_invalid_ratio_gat`: Ratio of valid to invalid triples for GAT training.
`--drop_GAT`: Dropout probability for SpGAT layer.
`--entity_out_dim`: Entity output embedding dimensions.
`--out_long_dim`: Longer output embedding dimensions.
`--out_short_dim`: Shorter output embedding dimensions.
`--margin`: Margin used in hinge loss.


The following parameters are related to the GAT model. We borrow the
hyperparameters from the original paper.

`--data`: Specify the folder name of the dataset.

`--epochs_gat`: Number of epochs for gat training.

`--epochs_conv`: Number of epochs for convolution training.

`--lr`: Initial learning rate.

`--weight_decay_gat`: L2 reglarization for gat.

`--weight_decay_conv`: L2 reglarization for conv.

`--get_2hop`: Get a pickle object of 2 hop neighbors.

`--use_2hop`: Use 2 hop neighbors for training.  

`--partial_2hop`: Use only 1 2-hop neighbor per node for training.

`--output_folder`: Path of output folder for saving models.

`--batch_size_gat`: Batch size for gat model.

`--valid_invalid_ratio_gat`: Ratio of valid to invalid triples for GAT training.

`--drop_gat`: Dropout probability for attention layer.

`--alpha`: LeakyRelu alphas for attention layer.

`--nhead_GAT`: Number of heads for multihead attention.

`--margin`: Margin used in hinge loss.

`--batch_size_conv`: Batch size for convolution model.

`--alpha_conv`: LeakyRelu alphas for conv layer.

`--valid_invalid_ratio_conv`: Ratio of valid to invalid triples for conv training.

`--out_channels`: Number of output channels in conv layer.

`--drop_conv`: Dropout probability for conv layer.

### Reproducing results
Structure
```
├── checkpoints
│   ├── fb
│   │   └── out
│   │       ├── classification_result
│   │       ├── connectE2T.trained_present.pth
│   │       ├── connectE2T_TRT.trained_present.pth
│   │       ├── connectTRT.trained_present.pth
│   │       ├── conv
│   │       ├── rngat.trained_present.pth
│   │       ├── trained_2999.pth
│   │       └── trained_present.pth
│   ├── kinship
│   │   └── out
│   │       └── conv
│   ├── nell
│   │   └── out
│   │       └── conv
│   ├── umls
│   │   └── out
│   │       └── conv
│   ├── wn
│   │   └── out
│   │       └── conv
│   └── yago
│       └── out
│           └── classification_result
├── connect_e.log
├── create_batch.py 
├── create_batch_e2t.py
├── create_dataset_files.py
├── data
│   ├── FB15k
│   │   ├── fb15k_all.zip
│   └── YAGO
│       └── yago_all.zip
├── evaluate_anxiang.py
├── evaluate_anxiang_yago.py
├── evaluator_anxiangz.py
├── layers.py
├── layers_e2t.py
├── main.py  # legacy
├── main_2GAT.py # legacy
├── main_long2short.py # for fb15k
├── main_long2short_yago.py # for yago43k
├── models.py  # legacy
├── models_2GAT.py # legacy
├── models_e2t.py
├── models_long2short.py 
├── prepare.sh
├── preprocess.py
├── pytorch35.yml
└── utils.py
```
To reproduce the results published in the paper:      
When running for first time, run preparation script with:

        $ sh prepare.sh


* **Fb15k**
  Note: I hard code some directories in the source code, e.g., the name of the
  saving model. So when you resume your training, there will be some errors.
        $ python3 main_long2short.py --data ./data/FB15k/ --epochs_gat 3000 --output_folder .checkpoints/fb/out/ # train the GAT model

        $ python3 evaluate_anxiang.py --data ./data/FB15k/ --epochs_gat 3000 --output_folder .checkpoints/fb/out/ # train the decoder

### Citation
This paper is currently under review. We will release the citation link once it
gets accepted.
