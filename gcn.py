import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import scipy.sparse as sp
import numpy as np

def sparse_dense_mul(a,b,c):
    i=a._indices()[1]
    j=a._indices()[0]
    v=a._values()
    newv=(b[i]+c[j]).squeeze()
    newv=torch.exp(F.leaky_relu(newv))
    
    new=torch.sparse.FloatTensor(a._indices(), newv, a.size())
    return new
    
class SpecialSpmmFunction(torch.autograd.Function):
    """Special function for only sparse region backpropataion layer."""
    @staticmethod
    def forward(ctx, indices, values, shape, b):
        assert indices.requires_grad == False
        a = torch.sparse_coo_tensor(indices, values, shape)
        ctx.save_for_backward(a, b)
        ctx.N = shape[0]
        return torch.matmul(a, b)

    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors
        grad_values = grad_b = None
        if ctx.needs_input_grad[1]:
            grad_a_dense = grad_output.matmul(b.t())
            edge_idx = a._indices()[0, :] * ctx.N + a._indices()[1, :]
            grad_values = grad_a_dense.view(-1)[edge_idx]
        if ctx.needs_input_grad[3]:
            grad_b = a.t().matmul(grad_output)
        return None, grad_values, None, grad_b


class SpecialSpmm(nn.Module):
    def forward(self, indices, values, shape, b):
        return SpecialSpmmFunction.apply(indices, values, shape, b)

class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features,activation=None,dropout=False):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.ll=nn.Linear(in_features,out_features).cuda()
        
        self.activation=activation
        self.dropout=dropout
    def forward(self,x,adj,dropout=0):
        
        x=self.ll(x)
        x=torch.spmm(adj,x)
        if not(self.activation is None):
            x=self.activation(x)
        if self.dropout:
            x=F.dropout(x,dropout)
        return x
class TAGraph(nn.Module):
    def __init__(self, in_features, out_features,k=2,activation=None,dropout=False,norm=False):
        super(TAGraph, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.lin=nn.Linear(in_features*(k+1),out_features).cuda()
        self.norm=norm
        self.norm_func=nn.BatchNorm1d(out_features,affine=False)
        self.activation=activation
        self.dropout=dropout
        self.k=k
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.lin.weight, gain=gain)
    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.lin.weight, gain=gain)
        
    def forward(self,x,adj,dropout=0):
        
        fstack=[x]
        
        for i in range(self.k):
            y=torch.spmm(adj,fstack[-1])
            fstack.append(y)
        x=torch.cat(fstack,dim=-1)
        x=self.lin(x)
        if self.norm:
            x=self.norm_func(x)
        if not(self.activation is None):
            x=self.activation(x)
        if self.dropout:
            x=F.dropout(x,dropout)
        return x
    
class MLPLayer(nn.Module):

    def __init__(self, in_features, out_features,activation=None,dropout=False):
        super(MLPLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.ll=nn.Linear(in_features,out_features).cuda()
        
        self.activation=activation
        self.dropout=dropout
    def forward(self,x,adj,dropout=0):
        
        x=self.ll(x)
      #  x=torch.spmm(adj,x)
        if not(self.activation is None):
            x=self.activation(x)
        if self.dropout:
            x=F.dropout(x,dropout)
        return x
class GAThead(nn.Module):
    def __init__(self, in_features, dims,activation=F.leaky_relu):
        super(GAThead, self).__init__()
        self.in_features = in_features
        self.dims = dims
        self.ll=nn.Linear(in_features,dims).cuda()
        # here we get of every number
        self.ll_att=nn.Linear(dims,1).cuda()
        #self.ll_att2=nn.Linear(dims,1).cuda()
        self.special_spmm = SpecialSpmm()
        self.activation=activation
       
    def forward(self,x,adj,dropout=0):
        x=F.dropout(x,dropout)
        x=self.ll(x)
        value=self.ll_att(x)
        #value2=self.ll_att2(x)
        value=F.leaky_relu(value)
        value=20-F.leaky_relu(20-value)
        #print(value.max())
        value=torch.exp(value)
        #print(value.max())
       # value=sparse_dense_mul(adj,value,value2)
        
        #dividefactor=torch.sparse.sum(value,dim=1).to_dense().unsqueeze(1)
        
        dividefactor=torch.spmm(adj,value)
        #print(dividefactor.max(),dividefactor.min())
        x=x*value
        x=torch.spmm(adj,x)
        #print(x.shape,dividefactor.shape)
        #print((x!=x).sum())
        #print((dividefactor!=dividefactor).sum())
        x=x/dividefactor
        #print((x!=x).sum())
        if self.activation!=None:
            x=self.activation(x)
        return x

class GATLayer(nn.Module):
    def __init__(self, in_features, n_heads,dims,activation=None,type=0):
        super(GATLayer, self).__init__()
        self.in_features = in_features
        self.n_heads=n_heads
        self.dims = dims
        self.heads=nn.ModuleList()
        self.type=type
        for i in range(n_heads):
            self.heads.append(GAThead(in_features,dims,activation=activation))
            
        
    def forward(self,x,adj,dropout=0):
        xp=[]
        for i in range(self.n_heads):
            xp.append(self.heads[i](x,adj,dropout))
        #n*8
        if self.type==0:
            sum=torch.cat(xp,1)
        else:
            sum=torch.sum(torch.stack(xp),0)/self.n_heads
        
        return sum
class GAT(nn.Module):
    def __init__(self,num_layers,num_heads,head_dim):
        super(GAT, self).__init__()
        self.num_layers=num_layers
        self.num_heads=num_heads
        self.head_dim=head_dim
        
        self.layers=nn.ModuleList()
        #print(num_layers)
        
        
        for i in range(num_layers):
            if i!=num_layers-1:
                self.layers.append(GATLayer(num_heads[i]*head_dim[i],num_heads[i+1],head_dim[i+1],activation=F.elu))
            else:
                self.layers.append(GATLayer(num_heads[i]*head_dim[i],num_heads[i+1],head_dim[i+1],type=1))
    def forward(self,x,adj,dropout=0):
        for layer in self.layers:
             x=layer(x,adj,dropout=dropout)
        # x=F.softmax(x, dim=-1)
        return x

class MLP(nn.Module):
    def __init__(self,num_layers,num_features):
        super(MLP, self).__init__()
        self.num_layers=num_layers
        self.num_features=num_features
        self.layers=nn.ModuleList()
        #print(num_layers)
        
        for i in range(num_layers):
            if i!=num_layers-1:
                self.layers.append(MLPLayer(num_features[i],num_features[i+1],activation=F.elu,dropout=True).cuda())
            else:
                self.layers.append(MLPLayer(num_features[i],num_features[i+1]).cuda())
        #print(self.layers)
        
    def forward(self,x,adj,dropout=0):
        x=x
        for layer in self.layers:
            x=layer(x,adj,dropout=dropout)
        x=F.softmax(x, dim=-1)
        return x
class GCN(nn.Module):
    def __init__(self,num_layers,num_features,activation=F.elu):
        super(GCN, self).__init__()
        self.num_layers=num_layers
        self.num_features=num_features
        self.layers=nn.ModuleList()
        #print(num_layers)
        
        for i in range(num_layers):
            
            if i!=num_layers-1:
                self.layers.append(GraphConvolution(num_features[i],num_features[i+1],activation=activation,dropout=True).cuda())
            else:
                self.layers.append(GraphConvolution(num_features[i],num_features[i+1]).cuda())
        #print(self.layers)
        
    def forward(self,x,adj,dropout=0):
        
        for layer in self.layers:
            x=layer(x,adj,dropout=dropout)
       # x=F.softmax(x, dim=-1)
        return x
       
class GCN_norm(nn.Module):
    def __init__(self,num_layers,num_features):
        super(GCN_norm,self).__init__()
        self.GCN=GCN(num_layers,num_features)
        #self.ln=nn.LayerNorm(100).cuda()
        self.num_layers=num_layers
        self.num_features=num_features
        self.layers=nn.ModuleList()
        #print(num_layers)
        
        for i in range(num_layers):
            self.layers.append(nn.LayerNorm(num_features[i]).cuda())
            if i!=num_layers-1:
                self.layers.append(GraphConvolution(num_features[i],num_features[i+1],activation=F.elu,dropout=True).cuda())
            else:
                self.layers.append(GraphConvolution(num_features[i],num_features[i+1]).cuda())
        #print(self.layers)
        
    def forward(self,x,adj,dropout=0,min=-1000,max=1000):
        x=torch.clamp(x,min,max)
        for i in range(len(self.layers)):
            if i%2==1:
                x=self.layers[i](x,adj,dropout=dropout)
            else:
                x=self.layers[i](x)
       # x=F.softmax(x, dim=-1)
        return x
    
class rep(nn.Module):
    def __init__(self,num_features):
        super(rep,self).__init__()
        mid=num_features
        #mid=int(np.sqrt(num_features))+1
        #self.ln=nn.LayerNorm(100).cuda()
        self.num_features=num_features
        self.lin1=nn.Linear(num_features,mid)
        self.lin2=nn.Linear(mid,num_features)
        self.ln=nn.LayerNorm(mid)
        #self.att=nn.Linear(num_features,1)
        self.activation1=F.relu
        self.activation2=F.sigmoid
        #gain = nn.init.calculate_gain('relu')
        #nn.init.xavier_normal_(self.lin.weight, gain=gain)
        #print(num_layers)
        
        #print(self.layers)
        
    def forward(self,x,adj):
        '''
        att=self.att(x)
        att=self.activation1(att)
        att=F.softmax(att,dim=0)
        #print(att.size())
        '''
        sumlines=torch.sparse.sum(adj,[0]).to_dense()
        allsum=torch.sum(sumlines)
        avg=allsum/x.size()[0]
        att=sumlines/allsum
        att=att.unsqueeze(1)
        #print(att.size())
        
        normx=x/sumlines.unsqueeze(1)
        avg=torch.mm(att.t(),normx)
        #avg=torch.mean(x,dim=0,keepdim=True)
        
        y=self.lin1(avg)
        y=self.activation1(y)
        y=self.ln(y)
        y=self.lin2(y)
        y=self.activation2(y)
        y=0.25+y*2
        
        #print(x.size(),avg.size())
        dimmin=normx-avg  #n*100
        dimmin=torch.sqrt(att)*dimmin
        rep=torch.mm(dimmin.t(),dimmin) # 100*100
        #ones=torch.ones(x.size()).cuda() #n*100
        #ones=torch.sum(ones,dim=0,keepdim=True)
        covariance=rep
        #conv=covariance.unsqueeze(0)
        q=torch.squeeze(y)
        qq=torch.norm(q)**2
        ls=covariance*q
        ls=ls.t()*q
        diag=torch.diag(ls)
        sumdiag=torch.sum(diag)
        sumnondiag=torch.sum(ls)-sumdiag
        loss=sumdiag-sumnondiag/self.num_features
        diagcov=torch.diag(covariance)
        sumdiagcov=torch.sum(diagcov)
        sumnondiagcov=torch.sum(covariance)-sumdiagcov
        lscov=sumdiagcov-sumnondiagcov/self.num_features
        k=loss/lscov
        k=k*self.num_features/qq
        if not(self.training):
            #print(ls)
            print(k)
        #print(y.shape)
        x=x*y
        
        #print((z-x).norm())
        return x,k

class nonelayer(nn.Module):
    def __init__(self):
        super(nonelayer,self).__init__()
    def forward(self,x):
        return x
class TAGCN(nn.Module):
    def __init__(self,num_layers,num_features,k):
        super(TAGCN, self).__init__()
        self.num_layers=num_layers
        self.num_features=num_features
        self.layers=nn.ModuleList()
            #print(num_layers)
            
        for i in range(num_layers):
            if i!=num_layers-1:
                self.layers.append(TAGraph(num_features[i],num_features[i+1],k,activation=F.leaky_relu,dropout=True).cuda())
            else:
                self.layers.append(TAGraph(num_features[i],num_features[i+1],k).cuda())
        #self.reset_parameters()
            #print(self.layers)
    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()
        
    def forward(self,x,adj,dropout=0,min=-1000,max=1000):
        #x=torch.clamp(x,min,max)
        #x=torch.atan(x)*2/np.pi
        for i in range(len(self.layers)):
            x=self.layers[i](x,adj,dropout=dropout)
           
        return x
class TArep(nn.Module):
    def __init__(self,num_layers,num_features,k):
        super(TArep, self).__init__()
        self.num_layers=num_layers
        self.num_features=num_features
        self.layers=nn.ModuleList()
            #print(num_layers)
            
        for i in range(num_layers):
            self.layers.append(rep(num_features[i]).cuda())
            if i!=num_layers-1:
                self.layers.append(TAGraph(num_features[i],num_features[i+1],k,activation=F.leaky_relu,dropout=True).cuda())
            else:
                self.layers.append(TAGraph(num_features[i],num_features[i+1],k).cuda())
            #print(self.layers)
        
    def forward(self,x,adj,dropout=0,min=-1000,max=1000):
        #x=torch.clamp(x,min,max)
        #x=torch.atan(x)*2/np.pi
        kk=0
        for i in range(len(self.layers)):
            
            if i%2==0:
                #x=self.layers[i](x)
                x,k=self.layers[i](x,adj)
                kk=k+kk
            else:
            #print(i,self.layers[i].lin.weight.norm(),x.shape)
                x=self.layers[i](x,adj,dropout=dropout)
            
        return x,kk
def GCNadj(adj,pow=-0.5):
    adj2=sp.eye(adj.shape[0])+adj
    for i in range(len(adj2.data)):
        if (adj2.data[i]>0 and adj2.data[i]!=1):
            adj2.data[i]=1
    adj2 = sp.coo_matrix(adj2)
    
    rowsum = np.array(adj2.sum(1))
    d_inv_sqrt = np.power(rowsum, pow).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    adj2=d_mat_inv_sqrt @ adj2 @ d_mat_inv_sqrt
    
    return adj2.tocoo()
