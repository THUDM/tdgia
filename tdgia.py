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
parser.add_argument('--models',nargs='+',default=['gcn_lm'])
parser.add_argument('--modelsextra',nargs='+',default=[])
parser.add_argument('--modelseval',nargs='+',default=['gcn_lm','graphsage_norm','sgcn','rgcn','tagcn','appnp','gin'])
#extra models are only used for label approximation.
parser.add_argument('--gpu',default='0')
parser.add_argument('--strategy',default='gia')
parser.add_argument('--test_rate',type=int,default=0)
parser.add_argument('--test',type=int,default=50000)
parser.add_argument('--lr',type=float,default=1)
parser.add_argument('--step',type=float,default=0.2)
parser.add_argument('--weight1',type=float,default=0.9)
parser.add_argument('--weight2',type=float,default=0.1)
parser.add_argument('--add_rate',type=float,default=1)
parser.add_argument('--scaling',type=float,default=1)
parser.add_argument('--opt',default="clip")
parser.add_argument('--add_num',type=float,default=500)
parser.add_argument('--max_connections',type=int,default=100)
parser.add_argument('--connections_self',type=int,default=0)

parser.add_argument('--apply_norm',type=int,default=1)
parser.add_argument('--load',default="default")
parser.add_argument('--sur',default="default")
#also evaluate on surrogate models
parser.add_argument('--save',default="default")
args=parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
if args.dataset=="aminer":
    adj=pkl.load(open("dset/testset_adj.pkl",'rb'))
    features=np.load("dset/features.npy")
    labels=np.load("dset/label_a.npy")
    test_index=range(len(features)-50000,len(features))
    val_index=range(580000,len(features)-50000)
    testlabels=labels[test_index]
    args.max_connections=88
    
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

if args.strategy=="fgsm":
    args.step=1
if args.strategy=="uniform":
    args.step=1
opt=args.opt
if args.strategy=="gia":
    opt="sin"

#force it to use "clip"
if args.opt=="fclip":
    opt="clip"
args.opt=opt
    
print("node num:",len(features))
print("edge num:",adj.sum())
print("val num:",len(val_index))
print("test num:",len(test_index))
print("feats num:",len(features[0]))
print("feat range:",features.min(),features.max())

testindex=test_index
total=len(features)
num_classes=np.max(testlabels)+1
num_features=features.shape[1]

if args.test_rate>0:
    args.test=int(args.test_rate*len(features)/100)
    
add=int(args.add_rate*0.01*total)
if args.add_num>0:
    add=args.add_num

def generateaddon(applabels,culabels,adj,origin,cur,testindex,addmax=500,num_classes=18,connect=65,sconnect=20,strategy='random'):
    # applabels: 50000
    # culabels: confidency of 50000*18
    weight1=args.weight1
    weight2=args.weight2
    newedgesx=[]
    newedgesy=[]
    newdata=[]
    thisadd=0
    num_test=len(testindex)
    import random
    if 'uniform' in strategy:
        for i in range(addmax):
            x=i+cur
            for j in range(connect):
                id=(x-origin)*connect+j
                id=id%len(testindex)
                y=testindex[id]
                newedgesx.extend([x,y])
                newedgesy.extend([y,x])
                newdata.extend([1,1])
        thisadd=addmax
        add1=sp.csr_matrix((thisadd,cur))
        add2=sp.csr_matrix((cur+thisadd,thisadd))
        adj=sp.vstack([adj,add1])
        adj=sp.hstack([adj,add2])
        adj.row=np.hstack([adj.row,newedgesx])
        adj.col=np.hstack([adj.col,newedgesy])
        adj.data=np.hstack([adj.data,newdata])
        return thisadd,adj
    if 'fgsm' in strategy:
        for i in range(addmax):
            islinked=np.zeros(len(testindex))
            for j in range(connect):
                x=i+cur
                
                yy=random.randint(0,num_test-1)
                while islinked[yy]>0:
                    yy=random.randint(0,num_test-1)
                
                y=testindex[yy]
                newedgesx.extend([x,y])
                newedgesy.extend([y,x])
                newdata.extend([1,1])
        thisadd=addmax
        add1=sp.csr_matrix((thisadd,cur))
        add2=sp.csr_matrix((cur+thisadd,thisadd))
        adj=sp.vstack([adj,add1])
        adj=sp.hstack([adj,add2])
        adj.row=np.hstack([adj.row,newedgesx])
        adj.col=np.hstack([adj.col,newedgesy])
        adj.data=np.hstack([adj.data,newdata])
        return thisadd,adj
    
        
    addscore=np.zeros(num_test)
    deg=np.array(adj.sum(axis=0))[0]+1.0
    normadj=GCNadj(adj)
    normdeg=np.array(normadj.sum(axis=0))[0]
    print(culabels[-1])
    for i in range(len(testindex)):
        it=testindex[i]
        label=applabels[it]
        score=culabels[it][label]+2
        addscore1=score/deg[it]
        addscore2=score/np.sqrt(deg[it])
        sc=weight1*addscore1+weight2*addscore2/np.sqrt(connect+sconnect)
        addscore[i]=sc
        
    
    # higher score is better
    sortedrank=addscore.argsort()
    sortedrank=sortedrank[-addmax*connect:]
    
    labelgroup=np.zeros(num_classes)
    #separate them by applabels
    labelil=[]
    for i in range(num_classes):
        labelil.append([])
    random.shuffle(sortedrank)
    for i in sortedrank:
        label=applabels[testindex[i]]
        labelgroup[label]+=1
        labelil[label].append(i)
    
    
        
    
    
    pos=np.zeros(num_classes)
    print(labelgroup)
    for i in range(addmax):
          #print(thisadd,labelnum)
        for j in range(connect):
            smallest=1
            smallid=0
            for k in range(num_classes):
                if len(labelil[k])>0:
                    if (pos[k]/len(labelil[k]))<smallest :
                        smallest=pos[k]/len(labelil[k])
                        smallid=k
            
            tu=labelil[smallid][int(pos[smallid])]
            
            #return to random
            #tu=sortedrank[i*connect+j]
            pos[smallid]+=1
            x=cur+i
            y=testindex[tu]
            newedgesx.extend([x,y])
            newedgesy.extend([y,x])
            newdata.extend([1,1])
                
            
    islinked=np.zeros((addmax,addmax))
    for i in range(addmax):
        j=np.sum(islinked[i])
        rndtimes=100
        while (np.sum(islinked[i])<sconnect and rndtimes>0):
            x=i+cur
            rndtimes=100
            yy=random.randint(0,addmax-1)
                
            while (np.sum(islinked[yy])>=sconnect or yy==i or islinked[i][yy]==1) and (rndtimes>0):
                yy=random.randint(0,addmax-1)
                rndtimes-=1
                    
            if rndtimes>0:
                y=cur+yy
                islinked[i][yy]=1
                islinked[yy][i]=1
                newedgesx.extend([x,y])
                newedgesy.extend([y,x])
                newdata.extend([1,1])
                
    thisadd=addmax
            
            
    print(thisadd,len(newedgesx))
    add1=sp.csr_matrix((thisadd,cur))
    add2=sp.csr_matrix((cur+thisadd,thisadd))
    adj=sp.vstack([adj,add1])
    adj=sp.hstack([adj,add2])
    adj.row=np.hstack([adj.row,newedgesx])
    adj.col=np.hstack([adj.col,newedgesy])
    adj.data=np.hstack([adj.data,newdata])
    return thisadd,adj
    
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
    
def getmultiresult(adj,features,models,modeltype,origin,testindex):
    
    pred=[]
    predb=[]
    iw=0
    with torch.no_grad():
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
            iza=F.softmax(iza,dim=-1)
            iw=iza+iw
            #pred.append(izb)
            #print(izb)
    
    
    surlabel=iw.argmax(-1).cpu().numpy()
    return surlabel


    
def trainaddon(adj,thisadd,model,modelrank,testlabels,feature,origin,testindex,num_features=100,strategy='random',reallabels=None,maxlim=1,lbth=4,opt="sin"):
    
    processed_gcn_adj=GCNadj(adj)
    processed_gc2_adj=GCNadj(adj,pow=-1)
    processed_nsage_adj=SAGEadj(adj,pow=0)

    gcn_adj=buildtensor(processed_gcn_adj)
    gc2_adj=buildtensor(processed_gc2_adj)
    dim_adj=buildtensor(processed_nsage_adj)
    from dgl import DGLGraph
    from dgl.transform import add_self_loop
    dim2_adj=DGLGraph(adj)
    dim2_adj=add_self_loop(dim2_adj).to('cuda')
    
    feature_origin=feature[:origin]
    import copy
    feature_added=feature[origin:].cpu().data.numpy()
    #revert it back
    
    
    if opt=="sin":
        feature_added=feature_added/maxlim
        feature_added=np.arcsin(feature_added)
    feature_added=Variable(torch.FloatTensor(feature_added).cuda(),requires_grad=True)
    add=torch.randn((thisadd,num_features)).cuda()
    if opt=="clip":
        add=add*maxlim*0.5
    if strategy=='random' :
        return add.cpu()
    if (strategy=="degms" and len(feature_added)+thisadd<args.add_num):
        return add.cpu()
    if strategy=='overwhelming':
        return torch.ones((thisadd,num_features))*10000
        
    
    import copy
    
    addontensor=Variable(add)
    addontensor.requires_grad=True
    optimizer=torch.optim.Adam([{'params':[addontensor,feature_added]}], lr=args.lr)
    optimizer.zero_grad()
    best_val=1
    testlabels=torch.LongTensor(testlabels).cuda()
    ep=args.epochs
    if (thisadd+1)*50+1<ep:
        ep=(thisadd+1)*50+1
    if strategy=="degms":
        ep=args.epochs*1.25
    #print(model,modelrank)
    for epoch in range(ep):
      
      i=epoch%(len(model))
      model[i].eval()
      adjtensor=gcn_adj
      feature_orc=feature_origin+0.0
    
      if modelrank[i] in ['graphsage_norm']:
        adjtensor=dim_adj
      if modelrank[i] in ['graphsage_max']:
        adjtensor=dim2_adj
      if modelrank[i] in ['rgcn']:
        adjtensor=(gcn_adj,gc2_adj)
      if opt=="sin":
        feature_add=torch.sin(feature_added)*maxlim
        addontenso=torch.sin(addontensor)*maxlim
      if opt=="clip":
        feature_add=torch.clamp(feature_added,-maxlim,maxlim)
        addontenso=torch.clamp(addontensor,-maxlim,maxlim)
      featuretensor=torch.cat((feature_orc,feature_add,addontenso),0)
        
      out1=model[i](featuretensor,adjtensor,dropout=0)
      testout1=out1[testindex]
      loss=nn.CrossEntropyLoss(reduction='none')
      l2=loss(testout1,testlabels)
      if opt=="sin":
        l2=F.relu(-l2+lbth)**2
      if opt=="clip":
        l2=-l2
      l=torch.mean(l2)#+torch.mean(l3)*25
        
        
      if epoch%75==0:
          fullac=0
          
          testoc=testout1.argmax(1)
          acc=(testoc==testlabels)
          acc=acc.sum()/(len(testindex)+0.0)
          acc=acc.item()
          
          print("epoch:",epoch,"loss:",l," acc_tag:",acc)
          
          if reallabels is not None:
            with torch.no_grad():
            #print(len(testoc),len(reallabels))
                tlabels=torch.LongTensor(reallabels).cuda()
                acc=(testoc==tlabels)
                acc=acc.sum()/(len(testindex)+0.0)
                acc=acc.item()
              
                print("epoch:",epoch,"loss:",l," acc_tag:",acc)
                curlabels=getresult(adj,featuretensor,model[i],modelrank[i])
                curlabels=curlabels.argmax(1)
                result=checkresult(curlabels,reallabels,origin,testindex)
                result2=checkresult(curlabels,testoc.data.cpu().numpy(),origin,testindex)
                print(result,result2)
            
          best_addon=featuretensor[origin:].cpu().data
    
      
      optimizer.zero_grad()
      l.backward()
      
      optimizer.step()
    return best_addon
    
num=0
models=[]
load_str=""
mdsrank=[]
num_models=len(args.models)
args.models.extend(args.modelsextra)
for name in args.models:
    exec("models.append("+name+"("+str(num_features)+','+str(num_classes)+").cuda())")
    dir=name+"_"+args.dataset+"/0"

    models[-1].load_state_dict(torch.load(dir))
    mdsrank.append(name)
same=[]


feature=torch.FloatTensor(features)
featurer=Variable(feature,requires_grad=True).cuda()
mds=[]
for j in range(len(models)):
    mds.append(models[j])
    
print(len(mds))

prlabel=getmultiresult(adj,featurer,mds,mdsrank,total,args.test)
print((prlabel[testindex]==testlabels).sum())

num=0
models=models[:num_models]
mds=mds[:num_models]
feature_origin=feature*1.0
while num<add:
    featurer=Variable(feature,requires_grad=True).cuda()
    # start with 0, to shape it.
    with torch.no_grad():
        curlabels=F.softmax(getresult(adj,featurer,models[0],mdsrank[0]),dim=1)
        #curlabels=F.one_hot(getresult(adj,featurer,models[0],mdsrank[0]).argmax(-1))
        for j in range(1,len(models)):
            curlabels+=F.softmax(getresult(adj,featurer,models[j],mdsrank[j]),dim=1)
            #curlabels+=F.one_hot(getresult(adj,featurer,models[j],mdsrank[j]).argmax(-1))
    curlabels=1/len(models)*curlabels
        
        
    addmax=int(add-num)
    if (addmax>add*args.step):
        addmax=int(add*args.step)
    thisadd,adj_new=generateaddon(prlabel,curlabels,adj,total,total+num,testindex,sconnect=args.connections_self,addmax=addmax,num_classes=num_classes,connect=args.max_connections,strategy=args.strategy)
    if thisadd==0:
        thisadd,adj_new=generateaddon(prlabel,curlabels,adj,total,total+num,testindex,sconnect=args.connections_self,addmax=addmax,num_classes=num_classes,connect=args.max_connections,strategy=arg.strategy)
    if num<add:
        num+=thisadd
        adj=adj_new
        print(thisadd,adj.shape)
        best_addon=trainaddon(adj,thisadd,mds,mdsrank,prlabel[testindex],featurer,total,testindex,num_features=num_features,strategy=args.strategy,reallabels=testlabels,maxlim=args.scaling,opt=args.opt)
           # print(best_addon)
        feature=torch.cat((feature_origin,best_addon),0)
    same=[]
    for i in range(len(models)):
        featurer=Variable(feature,requires_grad=True).cuda()
        curlabels=getresult(adj,featurer,models[i],mdsrank[i])
        curlabels=curlabels.argmax(1)
        result=checkresult(curlabels,testlabels,total,testindex)
        same.append(result)
        print(i,same)
        print("bb attack average:",np.average(same),"std:",np.std(same))
    

adj=adj.tocsr()
ad=adj[total:]
if args.save=="default":
    args.save=args.dataset+'_'+args.models[0]
ww=os.listdir()
if not(args.save in ww):
    os.mkdir(args.save)
    
pkl.dump(ad,open(args.save+"/adj.pkl","wb+"))
mr=feature.cpu()
mr=mr.numpy()
mr=mr[total:]
        
np.save(args.save+"/feature.npy",mr)
    
    #print(epoch,l)






    
    
    

