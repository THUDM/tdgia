import os
import numpy as np
import pickle as pkl
import scipy.sparse as sp
def loadogb(st,dir):
    process_dir=dir+"/processed"
    
    wi=os.listdir(process_dir)
    # signifies that the data is processed
    if 'adj.pkl' in wi:
        adj=pkl.load(open(process_dir+'/adj.pkl','rb'))
        feat=np.load(process_dir+'/feature.npy')
        label=np.load(process_dir+'/labels.npy')
        train_index=np.load(process_dir+'/train_index.npy')
        val_index=np.load(process_dir+'/val_index.npy')
        test_index=np.load(process_dir+'/test_index.npy')
        return adj,feat,label,train_index,val_index,test_index
    else:
        path=dir+"/raw"
        wt=[]
        feats=open(path+"/node-feat.csv")
        ff=feats.readlines()
        for i in range(len(ff)):
            spc=ff[i].split(',')
            
            nlist=list(map(lambda j:float(j),spc))
            wt.append(nlist)

        feat=np.array(wt)
        num_nodes=len(feat)
        feats.close()
        np.save(process_dir+'/feature.npy',feat)
        lbs=open(path+"/node-label.csv")
        lb=lbs.readlines()
        ll=[]
        for i in range(len(lb)):
            ll.append(int(lb[i]))
        labels=np.array(ll)
        np.save(process_dir+'/labels.npy',labels)
        lbs.close()
        edges=open(path+"/edge.csv")
        eds=sp.lil_matrix((num_nodes,num_nodes))
        edge=edges.readlines()
        nownum=0
        for line in edge:
            nownum+=1
            if nownum%200000==0:
                print(nownum)
            id1=int(line.split(',')[0])
            id2=int(line.split(',')[1])
            eds[id1,id2]=1
            eds[id2,id1]=1
        eds=eds.tocsr()
        pkl.dump(eds,open(process_dir+"/adj.pkl","wb+"))
        edges.close()
        if st=='arxiv':
            sp_path=dir+"/split/time"
        if st=='products':
            sp_path=dir+"/split/sales_ranking"
        train_index=[]
        t=open(sp_path+"/train.csv").readlines()
        for line in t:
            train_index.append(int(line))
        tr_index=np.array(train_index)
        np.save(process_dir+'/train_index.npy',tr_index)
        valid_index=[]
        v=open(sp_path+"/valid.csv").readlines()
        for line in v:
            valid_index.append(int(line))
        val_index=np.array(valid_index)
        np.save(process_dir+'/val_index.npy',val_index)
        test_index=[]
        te=open(sp_path+"/test.csv").readlines()
        for line in te:
            test_index.append(int(line))
        te_index=np.array(test_index)
        np.save(process_dir+'/test_index.npy',te_index)
        return eds,feat,labels,train_index,val_index,te_index
        
        
            
        
    
    
