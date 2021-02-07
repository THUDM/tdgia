from gcn import *
import pickle as pkl
from models_gcn import *
import scipy.sparse as sp
import numpy as np
import argparse
from ogb import *
import os
from torch.autograd import Variable
parser = argparse.ArgumentParser(description="Run SNE.")
parser.add_argument('--dataset',default='ogb_arxiv')
parser.add_argument('--epochs',type=int,default=2001)
parser.add_argument('--modelseval',nargs='+',default=['sgcn','rgcn','gcn_lm','graphsage_norm','tagcn','appnp','gin'])
parser.add_argument('--eval_data',default='ogb_gcn_lm')
#extra models are only used for label approximation.
parser.add_argument('--gpu',default='0')
parser.add_argument('--strategy',default='gia')
parser.add_argument('--test_rate',type=int,default=0)
parser.add_argument('--test',type=int,default=50000)
parser.add_argument('--lr',type=float,default=1)
parser.add_argument('--step',type=float,default=0.2)
parser.add_argument('--weight1',type=float,default=1)
parser.add_argument('--weight2',type=float,default=0)
parser.add_argument('--add_rate',type=float,default=1)
parser.add_argument('--scaling',type=float,default=1)
parser.add_argument('--add_num',type=float,default=500)
parser.add_argument('--max_connections',type=int,default=88)
parser.add_argument('--connections_self',type=int,default=0)

parser.add_argument('--apply_norm',type=int,default=1)
parser.add_argument('--load',default="default")
parser.add_argument('--sur',default="default")
#also evaluate on surrogate models
parser.add_argument('--save',default="default")
args=parser.parse_args()
def combine_features(adj,features,add_adj,add_features):
    if args.eval_data=="no":
        return adj,features
    nfeature=np.concatenate([features,add_features],0)
    
    total=len(features)
    adj_added1=add_adj[:,:total]
    adj=sp.vstack([adj,adj_added1])
   # print(adj_added.shape)
    
    adj=sp.hstack([adj,add_adj.transpose()])
    for i in range(len(adj.data)):
        if (adj.data[i]!=0) and (adj.data[i]!=1):
            adj.data[i]=1
    return adj.tocsr(),nfeature
    
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
if args.dataset=="aminer":
    adj=pkl.load(open("dset/testset_adj.pkl",'rb'))
    features=np.load("dset/features.npy")
    labels=np.load("dset/label_a.npy")
    
    testlabels=labels[-args.test:]
    test_index=range(len(features)-args.test,len(features))

if args.dataset=="reddit":
    adj=sp.load_npz("reddit_adj.npz")
    adj=adj+adj.transpose()
    data=np.load("reddit.npz")
    features=data['feats']
    features[:,:2]*=0.025 # scale them to usual range
    trainlabels=data['y_train']
    vallabels=data['y_val']
    testlabels=data['y_test']
    train_index=data['train_index']
    val_index=data['val_index']
    test_index=data['test_index']
    args.scaling=0.25
    
if 'ogb' in args.dataset:
    st=args.dataset.split('_')[-1]
    dir="../ogb/"+st
    adj,features,labels,train_index,val_index,test_index=loadogb(st,dir)
    trainlabels=labels[train_index]
    vallabels=labels[val_index]
    testlabels=labels[test_index]
if args.eval_data!='no':
    inc_adj=pkl.load(open(args.eval_data+"/adj.pkl",'rb'))
    inc_feat=np.load(args.eval_data+"/feature.npy")
else:
    inc_adj=0
    inc_feat=0
adj,features=combine_features(adj,features,inc_adj,inc_feat)

testindex=test_index
total=len(features)
num_classes=np.max(testlabels)+1
num_features=features.shape[1]


def getprocessedadj(adj,modeltype,feature=None):
    processed_adj=GCNadj(adj)
    if modeltype in ["graphsage_norm"]:
        processed_adj=SAGEadj(adj,pow=0)
    if modeltype=="graphsage_max":
        from dgl import DGLGraph
        from dgl.transform import add_self_loop
        dim2_adj=DGLGraph(adj)
        processed_adj=add_self_loop(dim2_adj).to('cuda')
    if modeltype=="rgcn":
        processed_adj=(processed_adj,GCNadj(adj,pow=-1))
    return processed_adj
    
def getprocessedfeat(feature,modeltype):
    feat=feature+0.0
    return feat
    
    
    
def getresult(adj,features,model,modeltype):
    processed_adj=getprocessedadj(adj,modeltype,feature=features.data.cpu().numpy())
    features=getprocessedfeat(features,modeltype)
    if modeltype=="graphsage_max":
        adjtensor=processed_adj
    if modeltype=="rgcn":
        adjtensor=(buildtensor(processed_adj[0]),buildtensor(processed_adj[1]))
    if not(modeltype in ["graphsage_max","rgcn"]):
        sparserow=torch.LongTensor(processed_adj.row).unsqueeze(1)
        sparsecol=torch.LongTensor(processed_adj.col).unsqueeze(1)
        sparseconcat=torch.cat((sparserow,sparsecol),1).cuda()
        sparsedata=torch.FloatTensor(processed_adj.data).cuda()
        adjtensor=torch.sparse.FloatTensor(sparseconcat.t(),sparsedata,torch.Size(processed_adj.shape)).cuda()
    feat=features+0
    model.eval()
    with torch.no_grad():
        result=model(features,adjtensor,dropout=0)
    return result

def checkresult(curlabels,testlabels,origin,testindex):
    evallabels=curlabels[testindex]
    tlabels=torch.LongTensor(testlabels).cuda()
    acc=(evallabels==tlabels)
    acc=acc.sum()/(len(testindex)+0.0)
    acc=acc.item()
    return acc
    
def buildtensor(adj):
    sparserow=torch.LongTensor(adj.row).unsqueeze(1)
    sparsecol=torch.LongTensor(adj.col).unsqueeze(1)
    sparseconcat=torch.cat((sparserow,sparsecol),1).cuda()
    sparsedata=torch.FloatTensor(adj.data).cuda()
    import copy
    adjtensor=torch.sparse.FloatTensor(sparseconcat.t(),sparsedata,torch.Size(adj.shape)).cuda()
    return adjtensor
    
def getmultiresult(adj,features,models,modeltype,origin,testindex,testlabels):
    
    pred=[]
    predb=[]
    iw=0
    with torch.no_grad():
        result_acc=[]
        for i in range(len(models)):
            processed_adj=getprocessedadj(adj,modeltype[i],feature=features.data.cpu().numpy())
            if not(modeltype[i] in ['graphsage_max','rgcn']):
                adjtensor=buildtensor(processed_adj)
            if modeltype[i] =='graphsage_max':
                adjtensor=processed_adj
            if modeltype[i]=='rgcn':
                adjtensor=(buildtensor(processed_adj[0]),buildtensor(processed_adj[1]))
            feat=getprocessedfeat(features,modeltype[i])
            models[i].eval()
            iza=models[i](feat,adjtensor,dropout=0)
           # izb=iza.argmax(-1).cpu().numpy()
            iza=iza.argmax(-1).cpu().numpy()
            sumsc=(iza[testindex]==testlabels).sum()
            print( modeltype[i],sumsc/len(testlabels))
            result_acc.append(sumsc/len(testlabels))
            
            
            #pred.append(izb)
            #print(izb)
        st=sorted(result_acc)
        weight=[0.3,0.24,0.18,0.12,0.08,0.05,0.03]
        ff=0
        for i in range(len(st)):
            ff+=st[-i-1]*weight[i]
        print("average:",np.average(st))
        print("3-max:",np.average(st[-3:]))
        print("w-average:",ff)
    
   
    return ff


    

num=0
models=[]
load_str=""
mdsrank=[]
num_models=len(args.modelseval)
for name in args.modelseval:
    exec("models.append("+name+"("+str(num_features)+','+str(num_classes)+").cuda())")
    dir=name+"_"+args.dataset+"/1"

    models[-1].load_state_dict(torch.load(dir))
    mdsrank.append(name)
same=[]


feature=torch.FloatTensor(features)
featurer=Variable(feature,requires_grad=True).cuda()
mds=[]
for j in range(len(models)):
    mds.append(models[j])
    
print(len(mds))

prlabel=getmultiresult(adj,featurer,mds,mdsrank,total,testindex,testlabels)

print(prlabel)
