import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import scipy.sparse as sp
import numpy as np

from gcn import *



class sglayer(nn.Module):
    def __init__(self,in_feat,out_feat):
        super(sglayer,self).__init__()
        self.lin=nn.Linear(in_feat,out_feat)
    def forward(self,x,adj,k=4):
        for i in range(k):
            x=torch.spmm(adj,x)
        x=self.lin(x)
        return x
        
class sgcn(nn.Module):
    def __init__(self,input_dim,output_dim,with_rep=False):
        super(sgcn,self).__init__()
        self.no=nn.BatchNorm1d(input_dim)
        self.in_conv=nn.Linear(input_dim,140)
        self.out_conv=nn.Linear(100,output_dim)
        self.act=torch.tanh
        self.layers=nn.ModuleList()
        self.with_rep=with_rep
        if with_rep:
            self.rep=nn.ModuleList()
            self.rep.append(rep(140))
            self.rep.append(rep(120))
            self.rep.append(rep(100))
        self.layers.append(sglayer(140,120))
        self.layers.append(nn.LayerNorm(120))
        self.layers.append(sglayer(120,100))
        self.layers.append(nn.LayerNorm(100))
        
        
    def forward(self,x,adj,dropout=0):
        x=self.no(x)
        x=self.in_conv(x)
        x=self.act(x)
        x=F.dropout(x,dropout)
        for i in range(len(self.layers)):
            
            if i%2==0:
                if self.with_rep:
                    x=self.rep[int(i/2)](x,adj)
                x=self.layers[i](x,adj)
            else:
                x=self.layers[i](x)
                x=self.act(x)
        if self.with_rep:
            x=self.rep[-1](x,adj)
        x=F.dropout(x,dropout)
        x=self.out_conv(x)
        
        return x
def SAGEadj(adj,pow=-1):
    adj2=sp.eye(adj.shape[0])*(1)+adj
    for i in range(len(adj2.data)):
        if (adj2.data[i]>0 and adj2.data[i]!=1):
            adj2.data[i]=1
        if (adj2.data[i]<0):
            adj2.data[i]=0
    adj2.eliminate_zeros()
    adj2 = sp.coo_matrix(adj2)
    if pow==0:
        return adj2.tocoo()
    rowsum = np.array(adj2.sum(1))
    d_inv_sqrt = np.power(rowsum, pow).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    adj2=d_mat_inv_sqrt @ adj2
    
    return adj2.tocoo()
class GCwithself(nn.Module):
    def __init__(self, in_features, out_features,activation=None,dropout=False):
        super(GCwithself, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.ll=nn.Linear(in_features,out_features).cuda()
        self.ll_self=nn.Linear(in_features,out_features).cuda()
        self.activation=activation
        self.dropout=dropout
    def forward(self,x,adj,dropout=0):
        x2=self.ll_self(x)
        x=self.ll(x)
        x=torch.spmm(adj,x)
        if not(self.activation is None):
            x=self.activation(x+x2)
        if self.dropout:
            x=F.dropout(x,dropout)
        return x
class tsail_sur(nn.Module):
    def __init__(self):
        super(tsail_sur,self).__init__()
        self.layers=nn.ModuleList()
        num_layers=3
        num_features=[100,200,128,128]
        for i in range(num_layers):
            self.layers.append(GCwithself(num_features[i],num_features[i+1],activation=F.relu,dropout=True).cuda())
            
        self.fc1 = nn.Linear(128, 128)
        self.fc2 = nn.Sequential(
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 18)
        )
    def forward(self,x,adj,dropout=0,min=0,max=0):
        
        for layer in self.layers:
            x=layer(x,adj,dropout=dropout)
        x=F.relu(self.fc1(x))
        x=self.fc2(x)
        return x
        
def tsail_pre(x):
    x=torch.clamp(x,-0.4,0.4)
    x.data[x.abs().ge(0.39).sum(1)>20]=0
    return x

class normSage(nn.Module):
    def __init__(self,in_features,pool_features,out_features,activation=None,dropout=False):
        super(normSage,self).__init__()
        self.pool_fc=nn.Linear(in_features,pool_features)
        self.fc1=nn.Linear(pool_features,out_features)
        self.fc2=nn.Linear(pool_features,out_features)
        self.activation=activation
        self.dropout=dropout
        
    def forward(self,x,adj,dropout=0,mu=2.0):
        
        x=F.relu(self.pool_fc(x))
        #usb=torch.max(x).data.item()
        
        #print(usb,usa)
        x3=x**mu
        x2=torch.spmm(adj,x3)**(1/mu)
        
        # In original model this is actually max-pool, but **10/**0.1 result in graident explosion. However we can still achieve similar performance using 2-norm.
        x4=self.fc1(x)
        x2=self.fc2(x2)
        x4=x4+x2
        if self.activation is not None:
            x4=self.activation(x4)
        if self.dropout:
            x4=F.dropout(x4,dropout)
        
        return x4
        
        
        
        
class graphsage_norm(nn.Module):
    def __init__(self,input_dim,output_dim,with_rep=False):
        super(graphsage_norm,self).__init__()
        self.layers=nn.ModuleList()
        num_layers=5
        num_features=[input_dim,70,70,70,70,output_dim]
        self.with_rep=with_rep
        if with_rep:
            self.rep=nn.ModuleList()
            
        for i in range(num_layers):
            if with_rep:
                self.rep.append(rep(num_features[i]))
            if i!=num_layers-1:
                self.layers.append(normSage(num_features[i],num_features[i],num_features[i+1],activation=F.relu,dropout=True))
            else:
                self.layers.append(normSage(num_features[i],num_features[i],num_features[i+1]))

    def forward(self,x,adj,dropout=0,min=0,max=0):
        #x=F.normalize(x,dim=1)
        for layer in self.layers:
            x=F.normalize(x,dim=1)
            x=layer(x,adj,dropout=dropout)
        return x
        
from dgl.nn.pytorch.conv import SAGEConv

class graphsage_max(nn.Module):
    def __init__(self,in_feats,out_feats):
        super(graphsage_max,self).__init__()
        self.layers=nn.ModuleList()
        num_layers=5
        num_features=[in_feats,70,70,70,70,out_feats]
        aggregator_type="pool"
        dropout=0.1
        for i in range(num_layers):
            if i!=num_layers-1:
                self.layers.append(SAGEConv(num_features[i],num_features[i+1],aggregator_type,activation=F.relu,feat_drop=dropout))
            else:
                self.layers.append(SAGEConv(num_features[i],num_features[i+1],aggregator_type,activation=None))

    def forward(self,x,adj,dropout=0):
        x=F.normalize(x,dim=1)
        for layer in self.layers:
            x=layer(adj,x)
        return x
        
        
class rgcn_conv(nn.Module):
    def __init__(self,in_feats,out_feats,act0=F.elu,act1=F.relu,initial=False,dropout=False):
        super(rgcn_conv,self).__init__()
        self.mean_conv=nn.Linear(in_feats,out_feats)
        self.var_conv=nn.Linear(in_feats,out_feats)
        self.act0=act0
        self.act1=act1
        self.initial=initial
        self.dropout=dropout
        
    def forward(self,mean,var=None,adj0=None,adj1=None,dropout=0):
        mean=self.mean_conv(mean)
        if self.initial:
            var=mean*1
        else:
            var=self.var_conv(var)
        mean=self.act0(mean)
        var=self.act1(var)
        attention=torch.exp(-var)
        
        mean=mean*attention
        var=var*attention*attention
        mean=torch.spmm(adj0,mean)
        var=torch.spmm(adj1,var)
        if self.dropout:
            mean=self.act0(mean)
            var=self.act0(var)
            mean=F.dropout(mean,dropout)
            var=F.dropout(var,dropout)
        return mean,var
        
class rgcn(nn.Module):
    def __init__(self,in_feats,out_feats):
        super(rgcn,self).__init__()
        self.layers=nn.ModuleList()
        num_layers=4
        self.act0=F.elu
        self.act1=F.relu
        for i in range(num_layers):
         
            if i==0:
                self.layers.append(rgcn_conv(in_feats,150,act0=self.act0,act1=self.act1,initial=True,dropout=True))
            if (i>0 and i<num_layers-1):
                self.layers.append(rgcn_conv(150,150,act0=self.act0,act1=self.act1,dropout=True))
            if i==num_layers-1:
                self.layers.append(rgcn_conv(150,out_feats,act0=self.act0,act1=self.act1))

    def forward(self,x,adj,dropout=0):
        adj0,adj1=adj
        mean=x
        var=x
        for layer in self.layers:
            mean,var=layer(mean,var=var,adj0=adj0,adj1=adj1,dropout=dropout)
        sample=torch.randn(var.shape).cuda()
        output=mean+sample*torch.pow(var,0.5)
        
        return output
def cccn_adj(adj,pow=-0.5):
    adj2=adj+0
    for i in range(len(adj2.data)):
        if (adj2.data[i]>0 and adj2.data[i]!=1):
            adj2.data[i]=1
    adj2 = sp.coo_matrix(adj2)
    
    rowsum = np.array(adj2.sum(1))
    d_inv_sqrt = np.power(rowsum, pow).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    adj2=adj2+sp.eye(adj.shape[0])
    adj2=d_mat_inv_sqrt @ adj2 @ d_mat_inv_sqrt
    
    return adj2.tocoo()

class cccn_sur(nn.Module):
    def __init__(self):
        super(cccn_sur,self).__init__()
        self.gcn=GCN(3,[100,128,128,18],activation=F.relu)
    def forward(self,x,adj,dropout=0):
        x=torch.clamp(x,-1.8,1.8)
        x=self.gcn(x,adj,dropout=dropout)
        return x

class gcn_lm(nn.Module):
    def __init__(self,in_feat,out_feat):
        super(gcn_lm,self).__init__()
        self.ln=nn.LayerNorm(in_feat)
        self.gcn=GCN(4,[in_feat,256,128,64,out_feat])
    def forward(self,x,adj,dropout=0):
        #x=torch.clamp(x,-1.8,1.8)
        x=self.ln(x)
        x=self.gcn(x,adj,dropout=dropout)
        return x
        
class gcn(nn.Module):
    def __init__(self,in_feat,out_feat):
        super(gcn,self).__init__()
        #self.ln=nn.LayerNorm(in_feat)
        self.gcn=GCN(4,[in_feat,256,128,64,out_feat])
    def forward(self,x,adj,dropout=0):
        x=self.gcn(x,adj,dropout=dropout)
        return x
def daftstone_pre(adj,features):
    w=np.abs(features)
    a=np.sum(w>0.5,axis=1)
    b=np.sum(w>0.3,axis=1)
    idx=(a>35)+(b>60)
    m1=np.max(w,axis=1)
    for j in range(2):
        idx1=np.argmax(w,axis=1)
        for i in range(idx1.shape[0]):
            w[i,idx1[i]]=0
    
    m2=np.max(w,axis=1)
    idx1=(m1-m2<=0.002)
    idx2=np.where(m1==0)[0]
    idx1[idx2]=False
    scale=1.5
    dispersion = np.load("max_dispersion.npy")
    idx3 = np.sum(np.abs(features) > dispersion * scale, axis=1) !=0
    idx = np.where(idx + idx1 + idx3)[0]
    flag=np.zeros((len(features)), dtype=np.int)
    if (len(idx) != 0):
        features[idx,] = 0
        flag[idx]=1
    adj=adj.tocoo()
    adj.data[flag[adj.row]==1]=0
    adj.data[flag[adj.col]==1]=0
    adj=GCNadj(adj)
    return adj
    
class daftstone_sur(nn.Module):
    def __init__(self):
        super(daftstone_sur,self).__init__()
        self.layers=nn.ModuleList()
        dims=[100,32,32,32,32]
        for i in range(4):
            if i>=2:
                self.layers.append(TAGraph(dims[i],dims[i+1],k=1,activation=F.leaky_relu,norm=True,dropout=True))
            else:
                self.layers.append(TAGraph(dims[i],dims[i+1],k=1,activation=F.leaky_relu,dropout=True))
        self.ll=nn.Linear(64,18)
        #self.norm=nn.BatchNorm1d(affine=False)
        
    def forward(self,x,adj,dropout=0):
        outputs=[]
        for i in range(4):
            x=self.layers[i](x,adj,dropout=dropout)
            outputs.append(x.unsqueeze(0))
        
        ot=torch.cat(outputs)
        ot1=torch.sum(ot,dim=0)*0.25
        ot2,ol2=torch.max(ot,dim=0)
        ot=torch.cat([ot1,ot2],dim=1)
        #print(ot1.shape,ot2.shape,ot.shape)
        ot=self.ll(ot)
        return ot
        
class msupsu_sur(nn.Module):
    def __init__(self):
        super(msupsu_sur,self).__init__()
        dims=[100,128,128,128,18]
        self.gcn=GCN(4,dims,activation=F.relu)
    def forward(self,x,adj,dropout=0):
        x=self.gcn(x,adj,dropout=dropout)
        return x
        
def msupsu_pre(adj,feature):
    adj=adj.tocoo()
    maxdis=0
    mindeg=0
    dell=0
    for i in range(len(adj.row)):
        node1=adj.row[i]
        node2=adj.col[i]
        feat1=feature[node1]
        feat2=feature[node2]
        dis=np.linalg.norm(feat1-feat2)
        if dis>2.6:
            adj.data[i]=0
            dell+=1
        if dis>maxdis:
            maxdis=dis
        dot=np.dot(feat1,feat2)
        dot=dot/np.linalg.norm(feat1)/np.linalg.norm(feat2)
        if dot<0.01:
            if adj.data[i]!=0:
                adj.data[i]=0
                dell+=1
            
        if dot<mindeg:
            mindeg=dot
    print(maxdis,mindeg,dell)
    adj=GCNadj(adj)
    return adj
    
def degrm(adj,thresholdmin=100,thresholdmax=100):
    adj2=sp.eye(adj.shape[0])*(-1)+adj
    for i in range(len(adj2.data)):
        if (adj2.data[i]>0 and adj2.data[i]!=1):
            adj2.data[i]=1
        if (adj2.data[i]<0):
            adj2.data[i]=0
    adj2.eliminate_zeros()
    adj2 = sp.coo_matrix(adj2)
    rowsum = np.array(adj2.sum(1))
    for i in range(len(adj2.data)):
        a=rowsum[adj2.row[i]]
        b=rowsum[adj2.col[i]]
        if (a>=thresholdmin and a<=thresholdmax) or (b>=thresholdmin and b<=thresholdmax):
            adj2.data[i]=0
    #adj2.eliminate_zeros()
    #adj2[:,np.where(rowsum>=thresholdmin and rowsum<=thresholdmax)]=0
   
    adj2.eliminate_zeros()
    
    return adj2.tocoo()
        
class nutrino_sur(nn.Module):
    def __init__(self):
        super(nutrino_sur,self).__init__()
        dims=[100,100,18]
        self.gcn=GCN(2,dims,activation=F.relu)
    def forward(self,x,adj,dropout=0):
        x=self.gcn(x,adj,dropout=dropout)
        return x
class gin_conv(nn.Module):
    def __init__(self,in_feat,out_feat,act=F.relu,eps=0,bn=True,dropout=True):
        super(gin_conv,self).__init__()
        self.nn1=nn.Linear(in_feat,out_feat)
        self.nn2=nn.Linear(out_feat,out_feat)
        self.act=act
        self.eps=torch.nn.Parameter(torch.Tensor([eps]))
        self.bn=bn
        if bn:
            self.norm=nn.BatchNorm1d(out_feat)
        self.dropout=dropout
    def forward(self,x,adj,dropout=0):
        y=torch.spmm(adj,x)
        x=y+(1+self.eps)*x
        x=self.nn1(x)
        x=self.act(x)
        x=self.nn2(x)
        if self.bn:
            x=self.norm(x)
        if self.dropout:
            x=F.dropout(x,dropout)
        return x
        
class gin(nn.Module):
    def __init__(self,in_feat,out_feat):
        super(gin,self).__init__()
        dims=[in_feat,144,144,144,144]
        self.layers=nn.ModuleList()
        for i in range(4):
            self.layers.append(gin_conv(dims[i],dims[i+1]))
        self.ll1=nn.Linear(144,144)
        self.ll2=nn.Linear(144,out_feat)
        self.act=F.relu
    def forward(self,x,adj,dropout=0):
        #x=torch.clamp(x,-1.74,1.63)
        for layer in self.layers:
            x=layer(x,adj,dropout=dropout)
        x=F.relu(self.ll1(x))
        x=F.dropout(x,dropout)
        x=self.ll2(x)
        return x
        
class SparseDropout(torch.nn.Module):
    def __init__(self):
        super(SparseDropout, self).__init__()
        # dprob is ratio of dropout
        # convert to keep probability
        self.kprob=1

    def forward(self, x,dprob=0.5):
        mask=((torch.rand(x._values().size())+(1-dprob)).floor()).type(torch.bool)
        rc=x._indices()[:,mask]
        val=x._values()[mask]*(1.0/(1.0001-dprob))
        return torch.sparse.FloatTensor(rc, val)
class appnp(nn.Module):
    def __init__(self,in_feats,out_feats):
        super(appnp,self).__init__()
        self.ll1=nn.Linear(in_feats,128)
        self.ll2=nn.Linear(128,out_feats)
        self.alpha=0.01
        self.drop=SparseDropout()
        self.act=F.relu
        self.K=10
    def forward(self,x,adj,dropout=0):
        # use proper modification on dropout places
        x=self.ll1(x)
        x=self.act(x)
        x=F.dropout(x,dropout)
        
        x=self.ll2(x)
        x=F.dropout(x,dropout)
        for i in range(self.K):
            ad=self.drop(adj,dprob=dropout)
            #rint(adj.shape,ad.shape)
            x=(1-self.alpha)*torch.spmm(ad,x)+self.alpha*x
        return x
        
class tarep_sur(nn.Module):
    def __init__(self):
        super(tarep_sur,self).__init__()
        self.tag=TArep(4,[602,128,128,128,41],3)
    def forward(self,x,adj,dropout=0,min=0,max=0):
        x=self.tag(x,adj,dropout=dropout)
        return x
        
class tagcn(nn.Module):
    def __init__(self,input_dim,output_dim,with_rep=False):
        super(tagcn, self).__init__()
        
        k=3
        num_features=[input_dim,128,128,128,output_dim]
        self.num_layers=num_layers=4
        self.layers=nn.ModuleList()
            #print(num_layers)
        self.with_rep=with_rep
        if with_rep:
            self.rep_layers=nn.ModuleList()
        for i in range(num_layers):
            if with_rep:
                self.rep_layers.append(rep(num_features[i]))
            if i!=num_layers-1:
                self.layers.append(TAGraph(num_features[i],num_features[i+1],k,activation=F.leaky_relu,dropout=True).cuda())
            else:
                self.layers.append(TAGraph(num_features[i],num_features[i+1],k).cuda())
        #self.reset_parameters()
            #print(self.layers
        
    def forward(self,x,adj,dropout=0):
        
        for i in range(len(self.layers)):
            if self.with_rep:
                x=self.rep_layers[i](x,adj)
            x=self.layers[i](x,adj,dropout=dropout)
        return x
        
class mixnet(nn.Module):
    def __init__(self,input_dim,output_dim,with_rep=False):
        super(mixnet, self).__init__()
        self.layers=nn.ModuleList()
        self.layers.append(MLPLayer(input_dim,128,activation=F.relu,dropout=True))
        self.layers.append(GraphConvolution(128,output_dim))
        self.with_rep=with_rep
        if with_rep:
            self.rep_layers=nn.ModuleList()
            self.rep_layers.append(rep(input_dim))
            self.rep_layers.append(rep(128))
            
    def forward(self,x,adj,dropout=0):
        
        for i in range(len(self.layers)):
            if self.with_rep:
                x=self.rep_layers[i](x,adj)
           
            x=self.layers[i](x,adj,dropout=dropout)
        return x


from dgl.nn.pytorch.conv import GATConv
from dgl import function as fn
from dgl.nn.pytorch import edge_softmax

class GATSample(nn.Module):

    def __init__(self,
                 in_feats):
        super(GATSample, self).__init__()
        self.attn_l=nn.Linear(in_feats,1)
        self.attn_r=nn.Linear(in_feats,1)
        self.reset_parameters()
    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.attn_l.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_r.weight, gain=gain)


    def forward(self, graph, feat):
        
        graph = graph.local_var()
    
        feat_src = self.attn_l(feat)
        feat_dst = self.attn_r(feat)
        graph.ndata['h'] = feat_dst
        graph.update_all(fn.copy_u('h', 'm'),
                        fn.sum('m', 'h'))
        feat_d = graph.ndata.pop('h')
        feat_d=F.relu(feat_d+feat_src)+1
        g_u=F.relu(feat_src)+1
        degree=graph.in_degrees().float().clamp(min=1)
        p1=torch.pow(degree,0.5)*feat_d.squeeze()*g_u.squeeze()
        p1=p1/torch.sum(p1)*int(len(feat)*0.25)
        #p1=torch.log(p1)
        p1=p1.unsqueeze(1)
        
        return p1

class ASLayer(nn.Module):

    def __init__(self,
                 in_feats,
                 out_feats,
                 activation=None):
        super(ASLayer, self).__init__()
        self.attn_l=nn.Linear(in_feats,1)
        self.attn_r=nn.Linear(in_feats,1)
        self.reset_parameters()
        self.ll=nn.Linear(in_feats,out_feats)
        self.activation=activation
    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.attn_l.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_r.weight, gain=gain)


    def forward(self, graph, feat,p):
        
        graph = graph.local_var()
        
        feat_src = self.attn_l(feat)
        
        feat_dst = self.attn_r(feat)
        #if self.train:
        p=p+0.0001
        if self.training:
            feat_dst=feat_dst-torch.log(p)
        feat1=self.ll(feat)
        graph.srcdata.update({'ft': feat1, 'el': feat_src})
        graph.dstdata.update({'er': feat_dst})
        # compute edge attention, el and er are a_l Wh_i and a_r Wh_j respectively.
        graph.apply_edges(fn.u_add_v('el', 'er', 'e'))
    
        #e=e/256
        p=p.squeeze()
        graphi=torch.multinomial(p,int(len(feat)*0.25))
        
        origin_adj=graph.adjacency_matrix
        
            
        e = F.relu(graph.edata.pop('e'))+1
        graph.edata.update({'e':e})
        suff=torch.zeros((feat.shape[0],1)).cuda()
        if self.training:
            #graph.adjacency_matrix=graph.adjacency_matrix[:graphi]
            suff[graphi]=10000
            graph.dstdata.update({'a2':suff})
            graph.apply_edges(fn.v_add_e('a2','e','e2'))
            e=graph.edata.pop('e2')
            
        
        graph.edata['a']=edge_softmax(graph,e)
        graph.update_all(fn.u_mul_e('ft', 'a', 'm'),
                         fn.sum('m', 'ft'))
        rst = graph.dstdata['ft']
        graph.adjacency_matrix=origin_adj
        # residual
        #rst=self.ll(rst)
        #mean=torch.mean(x,dim=0)
       # mean_support=1/
        if self.activation:
            rst = self.activation(rst)
        return rst
    




class mixgat(nn.Module):
    def __init__(self,input_dim,output_dim,with_rep=False):
        super(mixgat, self).__init__()
        self.layers=nn.ModuleList()
        self.layers.append(MLPLayer(input_dim,512,activation=F.relu,dropout=True))
        #self.ll1=nn.Linear(512,1).cuda()
        #self.ll2=nn.Linear(512,1).cuda()
        #self.sp=GATSample(input_dim)
        self.layers.append(GATConv(512,output_dim,8,feat_drop=0,attn_drop=0.5))
        self.with_rep=with_rep
        if with_rep:
            self.rep_layers=nn.ModuleList()
            self.rep_layers.append(rep(input_dim))
            self.rep_layers.append(rep(512))
            
    def forward(self,x,adj,dropout=0):
        
        for i in range(len(self.layers)):
            if self.with_rep:
                x=self.rep_layers[i](x)
            if i==0:
                x=self.layers[i](x,adj,dropout=dropout)
            else:
                #print(x.shape)
                #pwd=self.sp(adj,x)
                #print(pwd.shape)
                #print(p)
                #pwd=x*pwd
                x=self.layers[i](adj,x)
                x=torch.mean(x,dim=1)
                
        #print(x.shape)
        return x

class mixasgat(nn.Module):
    def __init__(self,input_dim,output_dim,with_rep=False):
        super(mixasgat, self).__init__()
        self.layers=nn.ModuleList()
        self.layers.append(MLPLayer(input_dim,512,activation=F.relu,dropout=True))
        #self.ll1=nn.Linear(512,1).cuda()
        #self.ll2=nn.Linear(512,1).cuda()
        self.sp=GATSample(input_dim)
        self.layers.append(ASLayer(512,output_dim))
        self.with_rep=with_rep
        if with_rep:
            self.rep_layers=nn.ModuleList()
            self.rep_layers.append(rep(input_dim))
            self.rep_layers.append(rep(512))
            
    def forward(self,x,adj,dropout=0):
        pwd=self.sp(adj,x)
        for i in range(len(self.layers)):
            if self.with_rep:
                x=self.rep_layers[i](x)
            if i==0:
                x=self.layers[i](x,adj,dropout=dropout)
            else:
                #print(x.shape)
                
                #print(pwd.shape)
                #print(p)
                #pwd=x*pwd
                x=self.layers[i](adj,x,pwd)
            
                #x=torch.mean(x,dim=1)
                
        #print(x.shape)
        return x
