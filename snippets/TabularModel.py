import torch
import torch.nn as tnn

class TabularModel(tnn.Module):

    """
    This class implements a generic ANN based on pytorch capable of receiving 
    both categorical and continuous inputs.
    The number of outputs and the structure of the network can be configured at start
    Adapted from the example provided in "Pytorch for Deep Learning" by J. Portilla @ Udemy
    """
    
    def __init__(self, emb_szs, n_cont, out_sz, layers, drop_p=0.5, activation_func=tnn.ReLU,batch_n=True):

        """
        The input variables are the following:
        emb_szs - describe the categorical variables as an array of tuples [(category_size,embedding_size),...]
        n_cont - total number of continuous variables
        out_sz - total number of outputs
        layers - number of perceptrons per layer defined as a list [n1,...]
        drop_p - dropout probability (<=0 == no dropout, >1 will be capped)
        batch_n - set to false if you don't want batch normalization
        activation_func- activation function
        """

        #call parent __init__
        super().__init__()

        #sum up total number of embeddings and total number of features
        self.n_emb = sum((nf for ni,nf in emb_szs))
        self.n_in = self.n_emb + n_cont

        #check sanity of dropout probability
        drop_p=min(drop_p,1)
        
        #set up the embedding and dropout for the categorical varables
        self.embeds   = tnn.ModuleList([tnn.Embedding(ni, nf) for ni,nf in emb_szs])
        self.emb_drop = tnn.Dropout(drop_p) if drop_p>0 else None

        #set up batch normalizaton of continuous variables
        self.bn_cont  = tnn.BatchNorm1d(n_cont) if batch_n else None
        
        # define the layer structure
        layerlist = []
        for i,n_out in enumerate(layers):
            n_in=self.n_in if i==0 else layers[i-1]
            layerlist.append(tnn.Linear(n_in,n_out)) 
            layerlist.append( activation_func(inplace=True) )
            if batch_n  : layerlist.append(tnn.BatchNorm1d(n_out))
            if drop_p>0 : layerlist.append(tnn.Dropout(drop_p))

        #last layer will produce the required outputs
        layerlist.append(tnn.Linear(layers[-1],out_sz))
        
        #list of layers is converted to a Sequential model as an attribute
        self.layers = tnn.Sequential(*layerlist)

        
    def forward(self, x_cat, x_cont):

        """
        extracts embedding values from categorical data, concatenates
        with the continuous values and performs the forward pass of the model
        """
        
        # extract embedding values from the incoming categorical data and perform dropout
        embeddings = []
        for i,emb in enumerate(self.embeds):
            embeddings.append( emb(x_cat[:,i]) )
        x_cat = torch.cat(embeddings, 1)
        if self.emb_drop:
            x_cat = self.emb_drop(x_cat)
        
        # normalize the incoming continuous data
        if self.bn_cont:
            x_cont = self.bn_cont(x_cont)

        #evaluate the sequential part of the model
        x = torch.cat([x_cat, x_cont], 1)
        x = self.layers(x)
        return x

        
