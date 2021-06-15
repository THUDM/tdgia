# TDGIA
code for 

``TDGIA:Effective Injection Attacks on Graph Neural Networks (KDD 2021, research track)``


1: Datasets, 

Reddit dataset (reddit_adj.npz, reddit.npz): from SNAP, (http://snap.stanford.edu/graphsage/), or from FastGCN https://github.com/matenure/FastGCN/issues/8#issuecomment-452463374

ogbn-arxiv dataset: from
https://snap.stanford.edu/ogb/data/nodeproppred/arxiv.zip

KDD-CUP( or aminer) dataset: from
https://github.com/THUDM/GIAAD

2: Code:
packages required: pytorch,dgl

To train gnns, use
``python train_gnns.py --model  $GNN   --gpu $gpu --dataset $dataset``

$GNN= rgcn (RobustGCN),sgcn, graphsage_norm (used in paper), gcn_lm, tagcn, appnp, gin

$dataset=aminer, reddit, ogb-arxiv

will automatically train 2 GNNs under the directory $GNN_$dataset  name of them will be 0 and 1.

To run TDGIA, use
``python tdgia.py --dataset $dataset --models $model --gpu $gpu --strategy $strategy ``

will generate its attack based on model $model_$dataset/0 ,  the generated attack will locate in $dataset_$model 

To evaluate, use 
``python GIA_evaluate.py --dataset $dataset --eval_data $path``

will evaluate this attack based on attack in package $path. Note that the all 7 models shall all be trained. The attack will based on model $model/1  (different from $model/0 which is used to generate attacks)

To evaluate on KDD-CUP using KDD-CUP defense submissions,check
https://github.com/THUDM/GIAAD,
and  copy the generated attack package to GIAAD/submission . 





 
