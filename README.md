# Boost-RS: Boosted Embeddings for Recommender Systems and its Application to Enzyme-Substrate Interaction Prediction
Recommender systems (RS),which are currently unexplored for the enzyme-substrate interaction prediction problem, can be utilizedto provide enzyme recommendations for substrates, and vice versa. The performance of Collaborative-Filtering (CF) recommender systems however hinges on the quality of embedding vectors of users anditems (enzymes and substrates in our case). Importantly, enhancing CF embeddings with auxiliary data,specially relational data (e.g., hierarchical or group labels), remains a challenge.

Boost-RS is an  innovative  general  RS  framework that  enhances RS performance by “boosting” embedding vectors based on auxiliary data. To incorporate this data, Boost-RSis trained on multiple relevant learning tasks and is dynamically tuned on the learning tasks. Boost-RSutilizes contrastive learning techniques to exploit relational data. To show the efficacy of Boost-RS for theenzyme-substrate prediction interaction problem, we apply the Boost-RS framework to several baselineCF models. When group attributes are rich in membership, we show that contrastive loss in the form oftriplet loss on group attributes is superior to utilizing the same data as individualized multi-label attributes. We also show that Boost-RS outperforms similarity-based models. 

## How to install
### Requirements: 
An [Anaconda python environment](https://www.anaconda.com/download) is recommmended.

```
conda create --name boost-rs --file enviroment.yml
source activate boost-rs
```

Check the environment.yml file, but primarily:
- python >= 3.5
- numpy
- scikit-learn
- scipy
- pytorch


## Example: Enzyme-Substrate Interaction Prediction

This repository contains an example of how to run the Boost-RS pipeline on the enzyme-substrate interaction prediction


## Components
- **mtl.py** -  functions for training and test on multi-task learning on auxiliary tasks using NMF baselines.
- **mtl-concat.py** - functions for training and test on using NMF concatenation of attributes baselines. 
- **util.py** - functions to for loading data.

## Data

- Data to run Boost-RS on enzyme-substrate interaction prediction.
Interaction data
Auxiliary data, including FP, RC, EC, and KO.

## Authors:
This script is written by Xinmeng Li, Li-ping Liu, Soha Hassoun (Soha.Hassoun@tufts.edu). 

Paper title: "Boosted Embeddings for Recommender Systems and its Application to Enzyme-Substrate Interaction Prediction"

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details

