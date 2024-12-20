import torch
import torch.nn as nn

from Ali_CCP.constants import vocabulary_size, column_type

from common.backbone_models import MultiLayerPerceptron, DeepFM, AutoInt, DCNv2

class CtrTower(nn.Module):
    """NN for CTR prediction"""

    def __init__(self, wide_input_dim, deep_input_dim, backbone_model, model_config):
        super().__init__()
        
        self.linear_model = nn.Linear(wide_input_dim, 1)  
        
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

    def forward(self, wide_inputs, deep_inputs):
      linear_out = self.linear_model(wide_inputs)
      deep_out = self.deep_model(deep_inputs)
      
      out = linear_out + deep_out
      
      return self.sigmoid(out)


class CvrTower(nn.Module):
    """NN for CTR prediction"""

    def __init__(self, wide_input_dim, deep_input_dim, backbone_model, model_config):
        super().__init__()
        
        self.linear_model_f = nn.Linear(wide_input_dim, 1)  
        self.linear_model_cf = nn.Linear(wide_input_dim, 1)  
        
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

    def forward(self, wide_inputs, deep_inputs):            
      linear_out_f = self.linear_model_f(wide_inputs)
      linear_out_cf = self.linear_model_cf(wide_inputs)
      
      deep_out_f = self.deep_model_f(deep_inputs)
      deep_out_cf = self.deep_model_cf(deep_inputs)
      
      out_f = self.sigmoid(linear_out_f + deep_out_f)
      out_cf = self.sigmoid(linear_out_cf + deep_out_cf)
      
      return out_f, out_cf


class DCMT(nn.Module):
    """Factual ESMM"""

    def __init__(self,
                 column_type: dict[str, list] = column_type,
                 feature_vocabulary: dict[str, int] = vocabulary_size,
                 embedding_size: int = 5,
                 backbone_model: str = 'mlp',
                 model_config: dict = {}):
      super().__init__()
      
      self.split_keys = column_type
      
      self.feature_vocabulary = feature_vocabulary
      self.feature_names = sorted(list(feature_vocabulary.keys()))

      self.embedding_size = embedding_size
      self.embedding_dict = nn.ModuleDict()
      self._init_weight()
      
      self.input_dim = len(feature_vocabulary) * embedding_size
      
      self.wide_input_dim = len(self.split_keys['wide']) * embedding_size
      self.deep_input_dim = self.input_dim - self.wide_input_dim
      
      self.ctr_model = CtrTower(self.wide_input_dim, self.deep_input_dim, backbone_model, model_config)
      self.cvr_model = CvrTower(self.wide_input_dim, self.deep_input_dim, backbone_model, model_config)
      
    def _init_weight(self):
      for name, size in self.feature_vocabulary.items():
        emb = nn.Embedding(size, self.embedding_size)
        nn.init.normal_(emb.weight, mean=0.0, std=0.01)
        self.embedding_dict[name] = emb

    def forward(self, inputs):
      # concat outputs of embedding layer
      wide_inputs = []
      deep_inputs = []
      
      for key, value in inputs.items():
        emb = self.embedding_dict[key](value)
        if key in self.split_keys['wide']:
          wide_inputs.append(emb)
        else:
          deep_inputs.append(emb)
      wide_input_vector = torch.cat(wide_inputs, axis=1)
      deep_input_vector = torch.cat(deep_inputs, axis=1)
          
      # Predict pCTR, pCVR
      pctr = self.ctr_model(wide_input_vector, deep_input_vector)
      pcvr_f, pcvr_cf = self.cvr_model(wide_input_vector, deep_input_vector)
      pctcvr = torch.mul(pctr, pcvr_f)
      
      output = {'pctr': pctr, 'pcvr': pcvr_f, 'pcvr_f': pcvr_f, 'pcvr_cf': pcvr_cf, 'pctcvr': pctcvr}
      
      return output
    