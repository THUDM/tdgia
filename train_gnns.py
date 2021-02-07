from gcn import *
import pickle as pkl
from models_gcn import *
import numpy as np
import argparse
import os
from torch.autograd import Variable
from ogb import loadogb
parser = argparse.ArgumentParser(description="Run SNE.")
parser.add_argument('--dataset',default='aminer')
parser.add_argument('--addon',default='no')
parser.add_argument('--epochs',type=int,default=10000)
parser.add_argument('--model',default='rgcn')
parser.add_argument('--dropout',type=float,default=0.1)
parser.add_argument('--lr',type=float,default=0.001)
parser.add_argument('--wd',type=float,default=0)
parser.add_argument('--opt',default="adam")
parser.add_argument('--gpu',default='0')
parser.add_argument('--train',type=int,default=580000)
parser.add_argument('--test',type=int,default=50000)
parser.add_argument('--train_rate',type=int,default=0)
parser.add_argument('--test_rate',type=int,default=0)
parser.add_argument('--delete_edge',type=int,default=0)
parser.add_argument('--load',default="")
parser.add_argument('--save',default="default")
parser.add_argument('--start',type=int,default=0)
args=parser.parse_args()

if args.dataset=="aminer":
    adj=pkl.load(open("dset/testset_adj.pkl",'rb'))
    features=np.load("dset/features.npy")
    labels=np.load("dset/label_a.npy")
    trainlabels=labels[0:args.train]
    vallabels=labels[args.train:-args.test]
    testlabels=labels[-args.test:]
    train_index=range(0,args.train)
    val_index=range(args.train,len(features)-args.test)
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
if 'ogb' in args.dataset:
    st=args.dataset.split('_')[-1]
    dir="../ogb/"+st
    adj,features,labels,train_index,val_index,test_index=loadogb(st,dir)
    trainlabels=labels[train_index]
    vallabels=labels[val_index]
    testlabels=labels[test_index]
    
def combine_features(adj,features,add_adj,add_features):
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
    
if args.addon!='no':
    inc_adj=pkl.load(open(args.addon+"/adj.pkl",'rb'))
    inc_feat=np.load(args.addon+"/feature.npy")
    adj,features=combine_features(adj,features,inc_adj,inc_feat)
    
#features=np.clip(features,-1,1)
#print(dims)
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

num_classes=np.max(trainlabels)+1
print(num_classes)
num_features=features.shape[1]

if args.train_rate>0:
    args.train=int(args.train_rate*len(features)/100)
if args.test_rate>0:
    args.test=int(args.test_rate*len(features)/100)

processed_adj=GCNadj(adj)
    
    
if args.model=='graphsage_norm':
    processed_adj=SAGEadj(adj,pow=0)
if args.model in ['graphsage_max']:
    from dgl import DGLGraph
    from dgl.transform import add_self_loop
    processed_adj=DGLGraph(adj)
    adjtensor=add_self_loop(processed_adj).to('cuda')
    
if args.model in ['rgcn']:
    processed_adj1=GCNadj(adj,pow=-1)

print("here3")
trainlabels=torch.LongTensor(trainlabels).cuda()
vallabels=torch.LongTensor(vallabels).cuda()
testlabels=torch.LongTensor(testlabels).cuda()
print("here4")

featuretens=torch.FloatTensor(features)
if args.model in ["tsail"]:
    featuretens=tsail_pre(featuretens)

print("here5")
#print(processed_adj.row)
if not(args.model in ['graphsage_max']):
    sparserow=torch.LongTensor(processed_adj.row).unsqueeze(1)
    sparsecol=torch.LongTensor(processed_adj.col).unsqueeze(1)
    sparseconcat=torch.cat((sparserow,sparsecol),1).cuda()
    sparsedata=torch.FloatTensor(processed_adj.data).cuda()

    adjtensor=torch.sparse.FloatTensor(sparseconcat.t(),sparsedata,torch.Size(processed_adj.shape)).cuda()

if args.model in ['rgcn']:
    sparsedata1=torch.FloatTensor(processed_adj1.data).cuda()
    adjtensor1=torch.sparse.FloatTensor(sparseconcat.t(),sparsedata1,torch.Size(processed_adj1.shape)).cuda()
    adjtensor=(adjtensor,adjtensor1)

print("here5")
wd=args.wd

    
best_val=0

if args.save=="default":
    args.save=args.model+'_'+args.dataset
    if args.addon!='no':
        args.save+='_poison'
    if args.delete_edge==1:
        args.save+="1"
if not (args.save in os.listdir()):
    os.mkdir(args.save)
#args.start=0
#model=TAGCN(len(dims)-1,dims,args.k)
for u in range(args.start,2):
    exec("model="+args.model+"("+str(num_features)+","+str(num_classes)+").cuda()")
    model.eval()
    if args.opt=="adam":
        optimizer=torch.optim.Adam(model.parameters(),lr=args.lr,weight_decay=wd)
    else:
        optimizer=torch.optim.SGD(model.parameters(), lr=args.lr,weight_decay=wd)
    optimizer.zero_grad()
    
    featuretensor=Variable(featuretens,requires_grad=True).cuda()
    
    best_val=0
    for epoch in range(args.epochs):
            #train
        #print(model.parameters().norm())
        #if args.model=="adversaries":
        #    out=model(featuretensor,adjtensor,adjtensor1,dropout=args.dropout)
       # else:
        model.train()
        out=model(featuretensor,adjtensor,dropout=args.dropout)
        kk=0
        if "rep" in args.model:
            out,kk=out
        trainout=out[train_index]
        
           # print(epoch,trainout,trainlabels)
        loss=nn.CrossEntropyLoss()
        l=loss(trainout,trainlabels)+kk*0.25
        optimizer.zero_grad()
        
        l.backward()
        #for i in model.layers:
        #    print(i.pool_fc.weight.grad.norm())
        optimizer.step()
           # print(epoch,l)
        if epoch%20==0:
            model.eval()
            with torch.no_grad():
                out=model(featuretensor,adjtensor,dropout=0)
                if "rep" in args.model:
                    out,kk=out
                valout=out[val_index].data
                
                valoc=valout.argmax(1)
                   # print(valoc,vallabels)
                acc=(valoc==vallabels).sum()
                acc=acc/(len(vallabels)+0.0)
                acc=acc.item()
                
                if best_val<acc:
                    best_val=acc
                    torch.save(model.state_dict(),args.save+"/"+str(u))
                    load=args.save+"/"+str(u)
                testout=out[test_index]
                #print(testout.norm())
                testoc=testout.argmax(1)
                test_acc=(testoc==testlabels).sum()
                test_acc=test_acc/(len(testlabels)+0.0)
                print("epoch:",epoch," acc:",acc," test_acc:",test_acc.item())

    print("no problem")
    model.load_state_dict(torch.load(load))
    
    model=model.cuda()
    model.eval()
 
  
    out=model(featuretensor,adjtensor,dropout=0)
    if "rep" in args.model:
        out,kk=out
    #out=model(featuretensor,adjtensor,dropout=0)
    testout=out[test_index]
    #testlabels=torch.LongTensor(testlabels).cuda()
    testoc=testout.argmax(1)
    test_acc=(testoc==testlabels).sum()
    test_acc=test_acc/(len(testlabels)+0.0)

    testo=testoc.data
    #pkl.dump(testoc,open("out.pkl","wb+"))

    print('test acc=',test_acc)

    del optimizer
    del model
    del out
    del trainout
    del valout
    del l
    del featuretensor
    
    torch.cuda.empty_cache()

    
    
    

