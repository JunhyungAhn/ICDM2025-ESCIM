import torch
import torch.nn as nn

from common.backbone_models import MultiLayerPerceptron, DeepFM, AutoInt, DCNv2
from common.layers import EmbeddingLayer
      
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
  """NN for CVR prediction"""

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

class Esmm(nn.Module):
    """Factual ESMM"""

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
      
      self.ctr_model = CtrTower(self.input_dim, backbone_model, model_config)
      self.cvr_model = CvrTower(self.input_dim, backbone_model, model_config)

    def forward(self, categorical_x, numerical_x):
      # concat outputs of embedding layer
      categorical_emb = self.embedding(categorical_x)
      numerical_emb = self.numerical_layer(numerical_x).unsqueeze(1)
      emb = torch.cat([categorical_emb, numerical_emb], 1).view(-1, self.embed_output_dim) 
          
      # Predict pCTR, pCVR
      pctr = self.ctr_model(emb)
      pcvr = self.cvr_model(emb)
      pctcvr = torch.mul(pctr, pcvr)
      
      output = {'pctr': pctr, 'pcvr': pcvr, 'pctcvr': pctcvr}
      
      return output
      
class CtrTower_C(nn.Module):
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
      input = inputs
      p = self.model(input)
      
      return self.sigmoid(p)
    
class CvrTower_C(nn.Module):
    """NN for CVR prediction"""

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

    def forward(self, inputs, click, z_v):            
      input = torch.cat([inputs, click, z_v], axis=1)
      p = self.model(input)
      
      return self.sigmoid(p)
    
class Esmm_C(nn.Module):
    """Counterfactual ESMM"""

    def __init__(
      self,
      categorical_field_dims, 
      numerical_num,
      embedding_size: int = 5,
      latent_dim: int = 5,
      backbone_model: str = 'mlp',
      model_config: dict = {}
    ):
    
      super().__init__()

      self.embedding_size = embedding_size
      
      self.embedding = EmbeddingLayer(categorical_field_dims, embedding_size)
      self.numerical_layer = torch.nn.Linear(numerical_num, embedding_size)
      self.embed_output_dim = (len(categorical_field_dims) + 1) * embedding_size
      
      self.input_dim = (len(categorical_field_dims) + 1) * embedding_size
      
      # input for ctr_network: inputs + z_c
      self.ctr_model = CtrTower_C(self.input_dim, backbone_model, model_config)
      # input for ctr_network: inputs + c + z_v
      self.cvr_model = CvrTower_C(self.input_dim+1+latent_dim, backbone_model, model_config)

    def forward(self, categorical_x, numerical_x, click, z_v):
      # Concat outputs of embedding layer
      categorical_emb = self.embedding(categorical_x)
      numerical_emb = self.numerical_layer(numerical_x).unsqueeze(1)
      emb = torch.cat([categorical_emb, numerical_emb], 1).view(-1, self.embed_output_dim) 
          
      # Predict pCTR, pCVR
      pctr = self.ctr_model(emb)
      pcvr = self.cvr_model(emb, click, z_v)
      pctcvr = torch.mul(pctr, pcvr)
      
      output = {'pctr': pctr, 'pcvr': pcvr, 'pctcvr': pctcvr}
      
      return output

# VAE model
class VAE(nn.Module):
    def __init__(
      self, 
      categorical_field_dims, 
      numerical_num,
      embedding_size: int = 5,
      latent_dim=20
    ):
    
      super().__init__()
      
      self.embedding_size = embedding_size
    
      self.embedding = EmbeddingLayer(categorical_field_dims, embedding_size)
      self.numerical_layer = torch.nn.Linear(numerical_num, embedding_size)
      self.embed_output_dim = (len(categorical_field_dims) + 1) * embedding_size
      
      self.input_dim = (len(categorical_field_dims) + 1) * embedding_size
      
      self.latent_dim = latent_dim
      
      self.encoder = nn.Sequential(
        nn.Linear(self.input_dim+2, 256),
        nn.ReLU(),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, latent_dim * 2)  # Two outputs for mean and log-variance
      )

      # Decoder
      self.decoder = nn.Sequential(
        nn.Linear(latent_dim, 128),
        nn.ReLU(),
        nn.Linear(128, 256),
        nn.ReLU(),
        nn.Linear(256, self.input_dim+2)
      )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, categorical_x, numerical_x, click, conversion):
      # Concat outputs of embedding layer
      categorical_emb = self.embedding(categorical_x)
      numerical_emb = self.numerical_layer(numerical_x).unsqueeze(1)
      emb = torch.cat([categorical_emb, numerical_emb], 1).view(-1, self.embed_output_dim)  
      
      x = torch.cat([emb, click, conversion], axis=1)
      
      # Encode
      enc_output = self.encoder(x)
      
      mu, logvar = enc_output[:, :self.latent_dim], enc_output[:, self.latent_dim:]

      # Reparameterize
      z = self.reparameterize(mu, logvar)

      # Decoder
      x_reconstructed = self.decoder(z)

      return x, x_reconstructed, mu, logvar
  