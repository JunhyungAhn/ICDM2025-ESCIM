import torch
import torch.nn as nn

from common.layers import EmbeddingLayer
from common.backbone_models import MultiLayerPerceptron, DeepFM, AutoInt, DCNv2
      
class CtrTower(nn.Module):
    """NN for CTR prediction"""

    def __init__(self, deep_input_dim, backbone_model, model_config):
        super().__init__()
        
        if backbone_model=='mlp':
          hidden_dims = model_config['mlp_dims']
          self.deep_model = MultiLayerPerceptron(deep_input_dim, hidden_dims)
        
        elif backbone_model=='deepfm':
          mlp_dims = model_config['mlp_dims']
          dropout = model_config['dropout']
          self.deep_model = DeepFM(deep_input_dim, mlp_dims, dropout)
        
        elif backbone_model=='autoint':
          num_heads = model_config['num_heads']
          num_layers = model_config['num_layers']
          dropout = model_config['dropout']
          self.deep_model = AutoInt(deep_input_dim, num_heads, num_layers, dropout)
        
        elif backbone_model=='dcnv2':
          mlp_dims = model_config['mlp_dims']
          cross_layer_num = model_config['cross_layer_num']
          dropout = model_config['dropout']
          self.deep_model = DCNv2(deep_input_dim, mlp_dims, cross_layer_num, dropout)
        
        else:
          raise Exception('Invalid Backbone Model')
        
        self.sigmoid = nn.Sigmoid()

    def forward(self, deep_inputs):
      deep_out = self.deep_model(deep_inputs)
      
      return self.sigmoid(deep_out)

class CvrTower(nn.Module):

    def __init__(self, deep_input_dim, backbone_model, model_config):
        super().__init__()
        
        if backbone_model=='mlp':
          hidden_dims = model_config['mlp_dims']
          self.deep_model_f = MultiLayerPerceptron(deep_input_dim, hidden_dims)
          self.deep_model_cf = MultiLayerPerceptron(deep_input_dim, hidden_dims)
        
        elif backbone_model=='deepfm':
          mlp_dims = model_config['mlp_dims']
          dropout = model_config['dropout']
          self.deep_model_f = DeepFM(deep_input_dim, mlp_dims, dropout)
          self.deep_model_cf = DeepFM(deep_input_dim, mlp_dims, dropout)
        
        elif backbone_model=='autoint':
          num_heads = model_config['num_heads']
          num_layers = model_config['num_layers']
          dropout = model_config['dropout']
          self.deep_model_f = AutoInt(deep_input_dim, num_heads, num_layers, dropout)
          self.deep_model_cf = AutoInt(deep_input_dim, num_heads, num_layers, dropout)
        
        elif backbone_model=='dcnv2':
          mlp_dims = model_config['mlp_dims']
          cross_layer_num = model_config['cross_layer_num']
          dropout = model_config['dropout']
          self.deep_model_f = DCNv2(deep_input_dim, mlp_dims, cross_layer_num, dropout)
          self.deep_model_cf = DCNv2(deep_input_dim, mlp_dims, cross_layer_num, dropout)
        
        else:
          raise Exception('Invalid Backbone Model')
        
        self.sigmoid = nn.Sigmoid()

    def forward(self, deep_inputs):            
  
      deep_out_f = self.deep_model_f(deep_inputs)
      deep_out_cf = self.deep_model_cf(deep_inputs)
      
      out_f = self.sigmoid(deep_out_f)
      out_cf = self.sigmoid( deep_out_cf)
      
      return out_f, out_cf

class DCMT(nn.Module):

    def __init__(
      self,
      categorical_field_dims, 
      numerical_num,
      embedding_size: int = 5,
      backbone_model: str = 'mlp',
      model_config: dict = {}
    ):

      super().__init__()
      
      self.embedding_size = embedding_size
      
      self.embedding = EmbeddingLayer(categorical_field_dims, embedding_size)
      self.numerical_layer = torch.nn.Linear(numerical_num, embedding_size)
      self.embed_output_dim = (len(categorical_field_dims) + 1) * embedding_size
      
      self.input_dim = (len(categorical_field_dims) + 1) * embedding_size
      
      self.deep_input_dim = self.input_dim
      
      self.ctr_model = CtrTower(self.deep_input_dim, backbone_model, model_config)
      self.cvr_model = CvrTower(self.deep_input_dim, backbone_model, model_config)

    def forward(self, categorical_x, numerical_x):
      # concat outputs of embedding layer
      categorical_emb = self.embedding(categorical_x)
      numerical_emb = self.numerical_layer(numerical_x).unsqueeze(1)
      emb = torch.cat([categorical_emb, numerical_emb], 1).view(-1, self.embed_output_dim) 
          
      # Predict pCTR, pCVR
      pctr = self.ctr_model(emb)
      pcvr_f, pcvr_cf = self.cvr_model(emb)
      pctcvr = torch.mul(pctr, pcvr_f)
      
      output = {'pctr': pctr, 'pcvr': pcvr_f, 'pcvr_f': pcvr_f, 'pcvr_cf': pcvr_cf, 'pctcvr': pctcvr}
      
      return output
    