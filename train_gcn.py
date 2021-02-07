from gcn import *
import pickle as pkl
from models_rep import *
from cs import *
import numpy as np
import argparse
import os
from torch.autograd import Variable
parser = argparse.ArgumentParser(description="Run SNE.")
parser.add_argument('--dataset',default='aminer')
parser.add_argument('--epochs',type=int,default=15000)
parser.add_argument('--model',default='tagcnr')
parser.add_argument('--dropout',type=float,default=0.1)
parser.add_argument('--lr',type=float,default=0.001)
parser.add_argument('--wd',type=float,default=0)
parser.add_argument('--opt',default="adam")
parser.add_argument('--gpu',default='0')
parser.add_argument('--train',type=int,default=580000)
parser.add_argument('--bc',type=int,default=0)
parser.add_argument('--test',type=int,default=50000)
parser.add_argument('--train_rate',type=int,default=0)
parser.add_argument('--test_rate',type=int,default=0)
parser.add_argument('--with_rep',type=int,default=0)
parser.add_argument('--load',default="")
parser.add_argument('--save',default="default")
parser.add_argument('--start',type=int,default=10)
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
    trainlabels=data['y_train']
    vallabels=data['y_val']
    testlabels=data['y_test']
    train_index=data['train_index']
    val_index=data['val_index']
    test_index=data['test_index']
    
#features=np.clip(features,-1,1)
#features=np.arctan(features)*2/np.pi

#print(dims)
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
inputdim=len(features[0])
num_classes=np.max(trainlabels)+1
trainlabels=torch.LongTensor(trainlabels).cuda()
vallabels=torch.LongTensor(vallabels).cuda()
testlabels=torch.LongTensor(testlabels).cuda()
print(num_classes)
num_features=features.shape[1]

def gen_tensor(processed_adj):
    sparserow=torch.LongTensor(processed_adj.row).unsqueeze(1)
    sparsecol=torch.LongTensor(processed_adj.col).unsqueeze(1)
    sparseconcat=torch.cat((sparserow,sparsecol),1).cuda()
    sparsedata=torch.FloatTensor(processed_adj.data).cuda()
    adjtensor=torch.sparse.FloatTensor(sparseconcat.t(),sparsedata,torch.Size(processed_adj.shape)).cuda()
    return adjtensor
tensor_gcn=gen_tensor(GCNadj(adj))
tensor_sage=gen_tensor(SAGEadj(adj))
    
if args.model in ["mixgat","mixasgat"]:
    from dgl import DGLGraph
    from dgl.transform import add_self_loop
    processed_adj=DGLGraph(adj)
    adjtensor=add_self_loop(processed_adj).to('cuda')
    


featuretens=torch.FloatTensor(features)

    
if not(args.model in ["mixgat","mixasgat"]):
    adjtensor=tensor_gcn


print("here5")
wd=args.wd

    
best_val=0

if args.save=="default":
    args.save=args.model
    if args.with_rep==1:
        args.save+="r"
   
if not (args.save in os.listdir()):
    os.mkdir(args.save)
ll=os.listdir(args.save)
if args.start>len(ll):
    args.start=len(ll)
#model=TAGCN(len(dims)-1,dims,args.k)

if args.with_rep==0:
    exec("model="+args.model+"("+str(inputdim)+","+str(num_classes)+").cuda()")
else:
    exec("model="+args.model+"("+str(inputdim)+","+str(num_classes)+",with_rep=True).cuda()")
        
model.eval()
if args.opt=="adam":
    optimizer=torch.optim.Adam(model.parameters(),lr=args.lr)
else:
    optimizer=torch.optim.SGD(model.parameters(), lr=args.lr,weight_decay=wd)
optimizer.zero_grad()
    
featuretensor=Variable(featuretens,requires_grad=True).cuda()
    
best_val=0
best_valn=np.zeros(20)
best_valn_test=np.zeros(20)
def get_norm(model):
    param=model.parameters()
    loss=0
    for i in param:
        loss=loss+torch.norm(i)
    return loss

def evaluation(out,labels):
    testoc=out.argmax(1)
    test_acc=(testoc==labels).sum()
    test_acc=test_acc/(len(labels)+0.0)
    test_acc=test_acc.item()
    #print("epoch:",epoch," acc:",acc," test_acc:",test_acc)
    return test_acc
thistrain=train_index
thislb=trainlabels
for epoch in range(args.epochs):
            #train
        #print(model.parameters().norm())
        #if args.model=="adversaries":
        #    out=model(featuretensor,adjtensor,adjtensor1,dropout=args.dropout)
       # else:
    
    if args.bc>0:
        curindex=np.random.choice(len(train_index),args.bc)
        thistrain=train_index[curindex]
        thislb=trainlabels[curindex]
    model.train()
    out=model(featuretensor,adjtensor,dropout=args.dropout)
    trainout=out[thistrain]
           # print(epoch,trainout,trainlabels)
    loss=nn.CrossEntropyLoss()
    l=loss(trainout,thislb)
    #l2=get_norm(model)*0.001
    #l=l+l2
    optimizer.zero_grad()
        
    l.backward()
        
    optimizer.step()
          
    if epoch%100==0:
        model.eval()
        out=model(featuretensor,adjtensor,dropout=0)
        out=F.softmax(out,dim=1).data
        valout=out[val_index]
        #print(valout)
        acc=evaluation(valout,vallabels)
        if best_val<acc:
            best_val=acc
            torch.save(model.state_dict(),args.save+"/"+"000")
            load=args.save+"/"+"000"
        
    
        testout=out[test_index]
        #print(testout,testlabels)
        test_acc=evaluation(testout,testlabels)
        
        
        outsource=correctstep(trainlabels,out,train_index,tensor_gcn,alpha=0.8)
        c_acc=evaluation(outsource[test_index].data,testlabels)
        
        
        outsource2=smoothstep(trainlabels,outsource,train_index,tensor_gcn,alpha=0.75)
        s_acc=evaluation(outsource2[test_index].data,testlabels)
        print("epoch:",epoch," acc:",acc," test_acc:",test_acc,'c_acc:',c_acc,'s_acc:',s_acc)
        if best_valn[-1]<acc:
            best_valn[-1]=acc
            best_valn_test[-1]=test_acc
            bts=len(best_valn)-1
            while (bts>0) and (best_valn[bts-1]<best_valn[bts]):
                sw=best_valn[bts-1]
                best_valn[bts-1]=best_valn[bts]
                best_valn[bts]=sw
                sw=best_valn_test[bts-1]
                best_valn_test[bts-1]=best_valn_test[bts]
                best_valn_test[bts]=sw
                bts-=1

print(np.average(best_valn_test),np.std(best_valn_test))
print("no problem")
model.load_state_dict(torch.load(load))
    
model=model.cuda()
model.eval()

out=model(featuretensor,adjtensor,dropout=0)
    #out=model(featuretensor,adjtensor,dropout=0)
testout=out[test_index]
    #testlabels=torch.LongTensor(testlabels).cuda()
testoc=testout.argmax(1)
test_acc=(testoc==testlabels).sum()
test_acc=test_acc/(args.test+0.0)

testo=testoc.data
    #pkl.dump(testoc,open("out.pkl","wb+"))

print('test acc=',test_acc)

    

