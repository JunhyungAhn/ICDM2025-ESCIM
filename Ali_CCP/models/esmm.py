import torch
import torch.nn as nn

from Ali_CCP.constants import vocabulary_size

from common.backbone_models import MultiLayerPerceptron, DeepFM, AutoInt, DCNv2

class CtrTower(nn.Module):
    """NN for CTR prediction"""

    def __init__(self, input_dim, backbone_model, model_config):
        super().__init__()
        if backbone_model=='mlp':
          mlp_dims = model_config['mlp_dims']
          self.model = MultiLayerPerceptron(input_dim, mlp_dims)
        
        elif backbone_model=='deepfm':
          mlp_dims = model_config['mlp_dims']
          dropout = model_config['dropout']
          self.model = DeepFM(input_dim, mlp_dims, dropout)
        
        elif backbone_model=='autoint':
          num_heads = model_config['num_heads']
          num_layers = model_config['num_layers']
          dropout = model_config['dropout']
          self.model = AutoInt(input_dim, num_heads, num_layers, dropout)
        
        elif backbone_model=='dcnv2':
          mlp_dims = model_config['mlp_dims']
          cross_layer_num = model_config['cross_layer_num']
          dropout = model_config['dropout']
          self.model = DCNv2(input_dim, mlp_dims, cross_layer_num, dropout)
        
        else:
          raise Exception('Invalid Backbone Model')
        
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs):
      p = self.model(inputs)
      
      return self.sigmoid(p)


class CvrTower(nn.Module):
    """NN for CTR prediction"""

    def __init__(self, input_dim, backbone_model, model_config):
        super().__init__()
        if backbone_model=='mlp':
          mlp_dims = model_config['mlp_dims']
          self.model = MultiLayerPerceptron(input_dim, mlp_dims)
        
        elif backbone_model=='deepfm':
          mlp_dims = model_config['mlp_dims']
          dropout = model_config['dropout']
          self.model = DeepFM(input_dim, mlp_dims, dropout)
        
        elif backbone_model=='autoint':
          num_heads = model_config['num_heads']
          num_layers = model_config['num_layers']
          dropout = model_config['dropout']
          self.model = AutoInt(input_dim, num_heads, num_layers, dropout)
        
        elif backbone_model=='dcnv2':
          mlp_dims = model_config['mlp_dims']
          cross_layer_num = model_config['cross_layer_num']
          dropout = model_config['dropout']
          self.model = DCNv2(input_dim, mlp_dims, cross_layer_num, dropout)
        
        else:
          raise Exception('Invalid Backbone Model')
        
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs):            
      # input = torch.cat([inputs], axis=1)
      p = self.model(inputs)
      
      return self.sigmoid(p)


class Esmm(nn.Module):
    """Factual ESMM"""

    def __init__(
      self,
      feature_vocabulary: dict[str, int] = vocabulary_size,
      embedding_size: int = 5,
      backbone_model: str = 'mlp',
      model_config: dict = {}
    ):
    
      super().__init__()
      
      self.feature_vocabulary = feature_vocabulary
      self.feature_names = sorted(list(feature_vocabulary.keys()))

      self.embedding_size = embedding_size
      self.embedding_dict = nn.ModuleDict()
      self._init_weight()
      
      self.input_dim = len(feature_vocabulary) * embedding_size
      
      self.ctr_model = CtrTower(self.input_dim, backbone_model, model_config)
      self.cvr_model = CvrTower(self.input_dim, backbone_model, model_config)
      
    def _init_weight(self):
      for name, size in self.feature_vocabulary.items():
        emb = nn.Embedding(size, self.embedding_size)
        nn.init.normal_(emb.weight, mean=0.0, std=0.01)
        self.embedding_dict[name] = emb

    def forward(self, inputs):
      # concat outputs of embedding layer
      feature_embedding = []
      for name in self.feature_names:
        embed = self.embedding_dict[name](inputs[name])
        feature_embedding.append(embed)
      feature_embedding = torch.cat(feature_embedding, axis=1)  
          
      # Predict pCTR, pCVR
      pctr = self.ctr_model(feature_embedding)
      pcvr = self.cvr_model(feature_embedding)
      pctcvr = torch.mul(pctr, pcvr)
      
      output = {'pctr': pctr, 'pcvr': pcvr, 'pctcvr': pctcvr}
      
      return output