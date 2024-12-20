import gc

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal

from torcheval.metrics import BinaryAUROC

from Ali_CCP.constants import vocabulary_size, column_type

from Ali_CCP.models.escim import Esmm_C, VAE
from Ali_CCP.models.esmm import Esmm
from Ali_CCP.models.escm2 import Escm2
from Ali_CCP.models.dcmt import DCMT

from common.loss import linear_annealing, vae_loss
from Ali_CCP.eval import eval_on_testset

def train_prior_model(config, train_loader, device):
  embedding_size = config["embedding_size"]
  latent_dim = len(vocabulary_size) * config["latent_dim_multiplier"]
  mlp_dims = config["mlp_dims"]
  lr = config["lr"]
  wd = config["wd"]
  
  normal_dist = Normal(0, 1)
  model = Esmm_C(
    feature_vocabulary=vocabulary_size,
    embedding_size=embedding_size,
    latent_dim=latent_dim,
    backbone_model='mlp',
    model_config = {
      'mlp_dims': mlp_dims
  })
  model = model.to(device)
  
  # Settings
  loss_fn = nn.BCELoss()
  weighted_loss_fn = nn.BCELoss(reduction='none')
  optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

  num_epochs = 2
  cvr_loss_weight = 0.1
  
  len_train_dataloader = len(train_loader)
  
  model.train()
  for epoch in range(num_epochs):
    running_total_loss = 0.0
    
    for _, (click, conversion, features) in enumerate(train_loader):
      for key in features.keys():
        features[key] = features[key].to(device)
      click = torch.unsqueeze(click.to(device), 1).to(torch.float)
      conversion = torch.unsqueeze(conversion.to(device), 1).to(torch.float)
      
      click_idx = (click==1).nonzero()[:, 0]
      
      # train with clicked data
      for key in features.keys():
        features[key] = features[key][click_idx].to(device)
      click = click[click_idx]
      conversion = conversion[click_idx]
      
      _batch_size = click.size(0)
      
      z_v = normal_dist.sample(sample_shape=(_batch_size, latent_dim))
      z_v = z_v.to(device)
      
      # Initialize gradient
      optimizer.zero_grad()
      
      # caluculate losses
      output = model(features, click, z_v)
      pctr, pcvr, pctcvr = output['pctr'], output['pcvr'], output['pctcvr']
      ctr_loss = loss_fn(pctr, click)
      ctcvr_loss = loss_fn(pctcvr, conversion)
      
      cvr_loss = weighted_loss_fn(pcvr, conversion)
      ctr_pred_clip = torch.clamp(pctr, min=0.05, max=1.0)
      cvr_loss = cvr_loss / ctr_pred_clip
      cvr_loss = torch.multiply(cvr_loss, click)
      cvr_loss = cvr_loss.mean()
      
      total_loss = ctr_loss + ctcvr_loss + cvr_loss_weight * cvr_loss
      
      # Backpropagation
      total_loss.backward()
      optimizer.step()

      running_total_loss += total_loss.item()

    running_total_loss /= len_train_dataloader

    txt = '\n'.join([
      f'Epoch={epoch}:',
      f' total_loss={running_total_loss:.5f},',
    ])
    print(txt)
    
  return model

def train_vae(config, train_loader, device):
  latent_dim = len(vocabulary_size) * config["latent_dim_multiplier"]
  lr = config["lr"]
  wd = config["wd"]
  
  vae = VAE(latent_dim=latent_dim).to(device) # input: inputs + click + conversion
  optimizer = optim.Adam(vae.parameters(), lr=lr, weight_decay=wd)
  num_epochs = 5

  len_train_dataloader = len(train_loader)
  
  vae.train()
  for epoch in range(num_epochs):
    total_loss = 0
    total_reconstruction_loss = 0
    total_kl_loss = 0
    
    for _, (click, conversion, features) in enumerate(train_loader):
      for key in features.keys():
        features[key] = features[key].to(device)
      click = torch.unsqueeze(click.to(device), 1).to(torch.float)
      conversion = torch.unsqueeze(conversion.to(device), 1).to(torch.float)
      
      click_idx = (click==1).nonzero()[:, 0]
      
      # only click label
      for key in features.keys():
        features[key] = features[key][click_idx].to(device)
      click = click[click_idx]
      conversion = conversion[click_idx]
        
      optimizer.zero_grad()
      
      batch, recon_batch, mu, logvar = vae(features, click, conversion)
      reconstruction_loss, kl_divergence = vae_loss(batch, recon_batch, mu, logvar)
      beta = linear_annealing(epoch, num_epochs, start_beta=0, end_beta=0.5)
      
      loss = reconstruction_loss + beta * kl_divergence
      
      loss.backward()
      optimizer.step()
      
      total_loss += loss.item()
      total_reconstruction_loss += reconstruction_loss.item()
      total_kl_loss += kl_divergence

    total_reconstruction_loss /= len_train_dataloader
    total_kl_loss /= len_train_dataloader

    txt = '\n'.join([
      f'Epoch={epoch}',
      f' reconstruction_loss={total_reconstruction_loss:.5f},',
      f' kl_divergence={total_kl_loss:.5f}'
      ])
    print(txt)
      
  return vae
  
def generate_cf_conv_label(method, click, conversion, cvr_cf):
  label = torch.zeros_like(cvr_cf)
  
  if method=='max':  
    f_cvr = cvr_cf[conversion==1]
    if f_cvr.numel()!=0:
      f_cvr_max = f_cvr.max()
      cf_label_idx = (cvr_cf>=f_cvr_max).nonzero()[:, 0]  
      if cf_label_idx.numel() != 0:
        for idx in cf_label_idx:
          label[idx] = 1.0
    
  elif method=='ratio':  
    num_click = click.sum().item()
    if num_click != 0:
      ratio = cvr_cf.size(0) // num_click
      k = int(conversion.sum().item() * ratio)  
      
      if k != 0:
        _, top_k_indices = torch.topk(cvr_cf.view(-1), k=k, dim=0)
        label[top_k_indices] = 1.0
        
  else:
    raise Exception('Invalid Label Transformation Method')

  return label

def train(config, train_loader, train_clicked_loader, val_loader, test_loader, device):
  embedding_size = config["embedding_size"]
  latent_dim = len(vocabulary_size) * config["latent_dim_multiplier"]
  batch_size = config["batch_size"]
  lr = config["lr"]
  wd = config["wd"]
  l2_reg = config["l2_reg"]
  
  backbone_model = config["backbone_model"]
  model_config = config["model_config"]
  
  cvr_f_loss_weight = config["cvr_f_loss_weight"]
  cvr_cf_loss_weight = config["cvr_cf_loss_weight"]
  
  train_models = config["train_models"]
  labeling_method = config["labeling_method"]

  num_trials = config["num_trials"]
  num_epochs = config["num_epochs"]
  valid_patience = config["valid_patience"]
  
  for trial in range(1, num_trials+1):
    print(f'********* Trial={trial} *********')
    
    for model in train_models:
      print(f'\n** Start training {model} model **')
      
      if model=="our":
        train_ours(
          config, 
          model,
          embedding_size, latent_dim, batch_size, lr, wd, l2_reg,
          cvr_f_loss_weight, cvr_cf_loss_weight,
          backbone_model, model_config, device,
          num_epochs, valid_patience, labeling_method,
          train_loader, train_clicked_loader, val_loader, test_loader
        )
        
      elif model in ["esmm", "escm2-ips", "escm2-dr", "dcmt", "multi-ips", "multi-dr"]:
        train_baseline(
          model, 
          embedding_size, lr, wd, l2_reg,
          backbone_model, model_config, device,
          num_epochs, valid_patience,
          train_loader, val_loader, test_loader
        )
  
      else:
        raise Exception('Invalid Model Name')
      

def train_ours(
  config, model_name, 
  embedding_size, latent_dim,  batch_size, lr, wd, l2_reg,
  cvr_f_loss_weight, cvr_cf_loss_weight,
  backbone_model, model_config, device,
  num_epochs, valid_patience, labeling_method,
  train_loader, train_clicked_loader, val_loader, test_loader
):

  prior_model = train_prior_model(config, train_clicked_loader, device)
  vae = train_vae(config, train_clicked_loader, device)

  labeling_method = config["labeling_method"]
  
  prior_model.eval()
  vae.eval()
  model = Esmm(
    feature_vocabulary=vocabulary_size,
    embedding_size=embedding_size,
    backbone_model=backbone_model,
    model_config=model_config
  ).to(device)
    
  # Settings
  loss_fn = nn.BCELoss()
  weighted_loss_fn = nn.BCELoss(reduction='none')
  optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

  best_val_auc = 0.0
  patience_count = 0
  
  valid_results_dict, test_results_dict = {}, {}
  best_perf_epoch = 1
  
  for epoch in range(1, num_epochs+1):
    print(f'\n== Epoch {epoch} ==')
    model.train()
    
    running_total_loss = 0.0

    for _, (click, conversion, features) in enumerate(train_loader):
      for key in features.keys():
        features[key] = features[key].to(device)
      click = torch.unsqueeze(click.to(device), 1).to(torch.float)
      conversion = torch.unsqueeze(conversion.to(device), 1).to(torch.float)

      # Initialize gradient
      optimizer.zero_grad()
      
      # Use only click label
      click_idx = (click==1).nonzero()[:, 0]
      features_clicked = features.copy()
      for key in features_clicked.keys():
        features_clicked[key] = features_clicked[key][click_idx].to(device)
      click_clicked = click[click_idx]
      conversion_clicked = conversion[click_idx]
      
      # Extract mu, sigma
      x, _, _, _ = vae(features_clicked, click_clicked, conversion_clicked)
      enc_output = vae.encoder(x)
      mu, logvar = enc_output[:, :latent_dim], enc_output[:, latent_dim:]
      std = torch.exp(0.5 * logvar)
      
      # Sampling from posterior distribution
      q_zv = Normal(mu.mean(dim=0), std.mean(dim=0))
      
      feature_embedding = []
      for name in prior_model.feature_names:
        embed = prior_model.embedding_dict[name](features[name])
        feature_embedding.append(embed)
      feature_embedding = torch.cat(feature_embedding, axis=1)  
      
      # Action: set click=1
      click_intv = torch.ones(size=(batch_size, 1)).to(device)
      z_v = q_zv.sample(sample_shape=(batch_size, )).to(device)
      
      # Prediction
      cvr_cf = prior_model.cvr_model(feature_embedding, click_intv, z_v)
      cf_conv_label = generate_cf_conv_label(labeling_method, click, conversion, cvr_cf)
      cf_conv_0_idx = (cf_conv_label==0).nonzero()[:, 0]
      cf_conv_1_idx = (cf_conv_label==1).nonzero()[:, 0]
      
      # Filter factual clicked indices
      cf_conv_0_idx = cf_conv_0_idx[~torch.isin(cf_conv_0_idx, click_idx)]
      cf_conv_1_idx = cf_conv_1_idx[~torch.isin(cf_conv_1_idx, click_idx)]
          
      # Generate Counterfactual Data
      features_cf = {key: value[cf_conv_0_idx] for key, value in features.items()}
      click_cf = torch.ones_like(cf_conv_0_idx).to(torch.float).reshape(-1, 1)
      conversion_cf = torch.zeros_like(cf_conv_0_idx).to(torch.float).reshape(-1, 1)
      
      if cf_conv_1_idx.numel() != 0:
        for key, value in features.items():
          features_cf[key] = torch.concat((features_cf[key], value[cf_conv_1_idx]), axis=0)
        click_cf = torch.concat((click_cf, torch.ones_like(cf_conv_1_idx).to(torch.float).reshape(-1, 1)), axis=0)
        conversion_cf = torch.concat((conversion_cf, torch.ones_like(cf_conv_1_idx).to(torch.float).reshape(-1, 1)), axis=0)
        
      # Calculate Factual Loss
      output = model(features)
      pctr, pcvr, pctcvr = output['pctr'], output['pcvr'], output['pctcvr']
      
      ctr_loss = loss_fn(pctr, click)
      ctcvr_loss = loss_fn(pctcvr, conversion)
      
      cvr_f_loss = weighted_loss_fn(pcvr, conversion)
      ctr_pred_clip = torch.clamp(pctr, min=0.05, max=1.0)
      cvr_f_loss = cvr_f_loss / ctr_pred_clip
      cvr_f_loss = torch.multiply(cvr_f_loss, click)
      cvr_f_loss = cvr_f_loss.mean()
      
      # Calculate Counterfactual Loss
      cvr_cf_loss = 0
      if features_cf is not None:  
        output_cf = model(features_cf)
        pctr_cf, pcvr_cf, _ = output_cf['pctr'], output_cf['pcvr'], output_cf['pctcvr']
          
        cvr_cf_loss = weighted_loss_fn(pcvr_cf, conversion_cf)
        ctr_pred_clip_cf = torch.clamp(pctr_cf, min=0.05, max=1.0)
        cvr_cf_loss = cvr_cf_loss / ctr_pred_clip_cf
        cvr_cf_loss = torch.multiply(cvr_cf_loss, click_cf)
        cvr_cf_loss = cvr_cf_loss.mean()
      
      # L2 Reg
      l2_reg_loss = torch.tensor(0., requires_grad=True)
      for name, param in model.named_parameters():
        if 'weight' in name:
          l2_reg_loss = l2_reg_loss + torch.norm(param, p=2)
      
      total_loss = ctr_loss + ctcvr_loss + cvr_f_loss_weight * cvr_f_loss + cvr_cf_loss_weight * cvr_cf_loss + l2_reg * l2_reg_loss
      
      total_loss.backward()
      optimizer.step()

      running_total_loss += total_loss.item()

    running_total_loss = running_total_loss / len(train_loader)
    print(f'Train_loss: {running_total_loss:.6f}')
    
    # Validation
    val_ctr_auc_value, val_cvr_auc_value, val_ctcvr_auc_value = valid(model_name, model, device, val_loader)
  
    valid_results_dict[epoch] = {
      'CTR_AUC': val_ctr_auc_value,
      'CVR_AUC': val_cvr_auc_value,
      'CTCVR_AUC': val_ctcvr_auc_value,
    }
    
    if val_ctcvr_auc_value >= best_val_auc:
      best_val_auc = val_ctcvr_auc_value
      
      patience_count = 0
      best_perf_epoch = epoch
      
    else:
      if patience_count==valid_patience-1:
        print("\n", "early stopped", "\n", sep="")
        break

      patience_count += 1
      print("\n", "patience count:\t", patience_count, "\n", sep="")
    
    # Test
    test_ctr_auc_value, test_cvr_auc_value, test_ctcvr_auc_value = eval_on_testset(model, device, test_loader)
    test_results_dict[epoch] = {
      'CTR_AUC': test_ctr_auc_value,
      'CVR_AUC': test_cvr_auc_value,
      'CTCVR_AUC': test_ctcvr_auc_value,
    }
  
  report_best_perf(valid_results_dict, test_results_dict, best_perf_epoch)


def train_baseline(
  model_name,
  embedding_size, lr, wd, l2_reg,
  backbone_model, model_config, device,
  num_epochs, valid_patience,
  train_loader, val_loader, test_loader
):
      
  if model_name in ["multi-ips", "esmm", "escm2-ips"]:
    model = Esmm(
      feature_vocabulary=vocabulary_size,
      embedding_size=embedding_size,
      backbone_model=backbone_model,
      model_config=model_config
    ).to(device)
  
  elif model_name in ["multi-dr", "escm2-dr"]:
    model = Escm2(
      feature_vocabulary=vocabulary_size,
      embedding_size=embedding_size,
      backbone_model=backbone_model,
      model_config=model_config,
      regularizer='dr'
    ).to(device)
  
  elif model_name=='dcmt':
    model = DCMT(
      column_type=column_type,
      feature_vocabulary=vocabulary_size,
      embedding_size=embedding_size,
      backbone_model=backbone_model,
      model_config=model_config
    ).to(device)
  
  else:
    raise Exception('Invalid backbone model')
      
  # Settings
  optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
  loss_fn = nn.BCELoss()
  weighted_loss_fn = nn.BCELoss(reduction='none')
  mse_loss_fn = torch.nn.MSELoss(reduction='none').to(device)
  
  cvr_loss_weight = 0.1
  cf_reg_weight = 0.001
  cvr_f_loss_weight = 1.0
  
  best_val_auc = 0.0
  patience_count = 0
  
  valid_results_dict, test_results_dict = {}, {}
  best_perf_epoch = 1
    
  for epoch in range(1, num_epochs+1):
    print(f'\n==== Epoch {epoch}')
    model.train()
    
    running_total_loss = 0.0
    for i, (click, conversion, features) in enumerate(val_loader):
      for key in features.keys():
        features[key] = features[key].to(device)
      click = torch.unsqueeze(click.to(device), 1).to(torch.float)
      conversion = torch.unsqueeze(conversion.to(device), 1).to(torch.float)

      # Initialize gradient
      optimizer.zero_grad()
            
      # caluculate losses
      output = model(features)
      
      if model_name in ["multi-ips", "esmm", "escm2-ips"]:
        pctr, pcvr, pctcvr = output['pctr'], output['pcvr'], output['pctcvr']
      elif model_name in ["multi-dr", "escm2-dr"]:
        pctr, pcvr, pctcvr, pimp = output['pctr'], output['pcvr'], output['pctcvr'], output['pimp']
      elif model_name=='dcmt':
        pctr, pcvr_f, pcvr_cf, pctcvr = output['pctr'], output['pcvr_f'], output['pcvr_cf'], output['pctcvr']
        
      ctr_loss = loss_fn(pctr, click)
      ctcvr_loss = loss_fn(pctcvr, conversion)
      
      # L2 Reg
      l2_reg_loss = torch.tensor(0., requires_grad=True)
      for name, param in model.named_parameters():
        if 'weight' in name:
          l2_reg_loss = l2_reg_loss + torch.norm(param, p=2)
      
      if model_name in ["multi-ips", "escm2-ips"]:
        cvr_loss = weighted_loss_fn(pcvr, conversion)
        ctr_pred_clip = torch.clamp(pctr, min=0.05, max=1.0)
        cvr_loss = cvr_loss / ctr_pred_clip
        cvr_loss = torch.multiply(cvr_loss, click)
        cvr_loss = cvr_loss.mean()
      
        if model_name=="escm2-ips":
          total_loss = ctr_loss + ctcvr_loss + l2_reg * l2_reg_loss + cvr_loss_weight * cvr_loss
        else:
          total_loss = cvr_loss
      
      elif model_name in ["multi-dr", "escm2-dr"]:
        squared_error = mse_loss_fn(pcvr, pimp)
        loss_imp = torch.multiply(squared_error, click)
        pctr_clip = torch.clamp(pctr, min=0.05, max=1.0)
        loss_imp = loss_imp / pctr_clip
        loss_imp = loss_imp.mean()

        error = (pcvr - pimp)
        loss_err = torch.multiply(error, click)
        loss_err = loss_err / pctr_clip + pimp
        loss_err = loss_err.mean()

        cvr_loss = loss_imp + loss_err
        
        if model_name=="escm2-dr":
          total_loss = ctr_loss + ctcvr_loss + l2_reg * l2_reg_loss + cvr_loss_weight * cvr_loss
        else:
          total_loss = cvr_loss
      
      elif model_name=='dcmt':
        # Factual Loss
        cvr_f_loss = weighted_loss_fn(pcvr_f, conversion)
        ctr_pred_clip = torch.clamp(pctr, min=0.05, max=1.0)
        cvr_f_loss = cvr_f_loss / ctr_pred_clip
        cvr_f_loss = torch.multiply(cvr_f_loss, click)
        cvr_f_loss = cvr_f_loss.mean()
        
        # Counterfactual Loss
        conversion_cf = torch.ones_like(pcvr_cf).to(torch.float).reshape(-1, 1)
        cvr_cf_loss = weighted_loss_fn(pcvr_cf, conversion_cf)
        ctr_pred_clip = torch.clamp(1-pctr, min=0.05, max=1.0)
        cvr_cf_loss = cvr_cf_loss / ctr_pred_clip
        cvr_cf_loss = torch.multiply(cvr_cf_loss, 1-click)
        cvr_cf_loss = cvr_cf_loss.mean()
        
        # Counterfactual Regularizer
        cf_reg_loss = (torch.abs(1 - (pcvr_f + pcvr_cf))).mean()
        
        cvr_loss = cvr_f_loss + cvr_cf_loss + cf_reg_weight * cf_reg_loss
        
        total_loss = ctr_loss + ctcvr_loss + l2_reg * l2_reg_loss + cvr_f_loss_weight * cvr_loss
      
      total_loss.backward()
      optimizer.step()

      running_total_loss += total_loss.item()

    running_total_loss = running_total_loss / len(train_loader)
    print(f'\nTrain_loss: {running_total_loss:.6f}')
    
    # Validation
    val_ctr_auc_value, val_cvr_auc_value, val_ctcvr_auc_value = valid(model_name, model, device, val_loader)
  
    valid_results_dict[epoch] = {
      'CTR_AUC': val_ctr_auc_value,
      'CVR_AUC': val_cvr_auc_value,
      'CTCVR_AUC': val_ctcvr_auc_value,
    }
    
    if val_ctcvr_auc_value >= best_val_auc:
      best_val_auc = val_ctcvr_auc_value
      
      patience_count = 0
      best_perf_epoch = epoch
      
    else:
      if patience_count == valid_patience - 1:
        print("\n", "early stopped", "\n", sep="")
        break

      patience_count += 1
      print("\n", "patience count:\t", patience_count, "\n", sep="")
    
    # Test
    test_ctr_auc_value, test_cvr_auc_value, test_ctcvr_auc_value = eval_on_testset(model, device, test_loader)
    test_results_dict[epoch] = {
      'CTR_AUC': test_ctr_auc_value,
      'CVR_AUC': test_cvr_auc_value,
      'CTCVR_AUC': test_ctcvr_auc_value,
    }
  
  report_best_perf(valid_results_dict, test_results_dict, best_perf_epoch)


def valid(model_name, model, device, val_loader):
  val_ctr_auc = BinaryAUROC().to('cpu')
  val_cvr_auc = BinaryAUROC().to('cpu')
  val_ctcvr_auc = BinaryAUROC().to('cpu')
  
  model.eval()
  for i, (click, conversion, features) in enumerate(val_loader):
    for key in features.keys():
      features[key] = features[key].to(device)
    click = torch.unsqueeze(click.to(device), 1).to(torch.float)
    conversion = torch.unsqueeze(conversion.to(device), 1).to(torch.float)
    
    click_idx = (click==1).to(torch.float32).nonzero()[:, 0]
    
    output = model(features)
    
    pctr, pctcvr = output['pctr'], output['pctcvr']
    if model_name=='dcmt':
      pcvr = output['pcvr_f']
    else:
      pcvr = output['pcvr']
      
    pcvr_clicked = pcvr[click_idx]
    conversion_clicked = conversion[click_idx]
    
    val_ctr_auc.update(pctr.squeeze().detach().cpu(), click.squeeze().detach().cpu())
    if pcvr_clicked.numel()!=1:
      val_cvr_auc.update(pcvr_clicked.squeeze().detach().cpu(), conversion_clicked.squeeze().detach().cpu())
    val_ctcvr_auc.update(pctcvr.squeeze().detach().cpu(), conversion.squeeze().detach().cpu())
    
    del features, click, conversion, output, pctr, pcvr, pctcvr

    if i % 100==0:
      gc.collect()
      torch.cuda.empty_cache()

  val_ctr_auc_value = val_ctr_auc.compute().to("cpu").item()
  val_cvr_auc_value = val_cvr_auc.compute().to("cpu").item()
  val_ctcvr_auc_value = val_ctcvr_auc.compute().to("cpu").item()

  txt = '\n'.join([
    f'Valid CTR AUC={val_ctr_auc_value:.5f},',
    f'Valid CVR AUC={val_cvr_auc_value:.5f},',
    f'Valid CTCVR AUC={val_ctcvr_auc_value:.5f}'
  ])
  print(txt)
  
  return val_ctr_auc_value, val_cvr_auc_value, val_ctcvr_auc_value

def report_best_perf(valid_results_dict, test_results_dict, best_perf_epoch):
  final_val_perf, final_test_perf = valid_results_dict[best_perf_epoch], test_results_dict[best_perf_epoch]
  val_ctr_auc_value, val_cvr_auc_value, val_ctcvr_auc_value = final_val_perf['CTR_AUC'], final_val_perf['CVR_AUC'], final_val_perf['CTCVR_AUC']
  test_ctr_auc_value, test_cvr_auc_value, test_ctcvr_auc_value = final_test_perf['CTR_AUC'], final_test_perf['CVR_AUC'], final_test_perf['CTCVR_AUC']
  
  txt = '\n'.join([
    '\n==== FINAL PERFORMANCE ====',
    f'Valid CTR AUC={val_ctr_auc_value:.6f},',
    f'Valid CVR AUC={val_cvr_auc_value:.6f},',
    f'Valid CTCVR AUC={val_ctcvr_auc_value:.6f}',
    f'Test CTR AUC={test_ctr_auc_value:.6f},',
    f'Test CVR AUC={test_cvr_auc_value:.6f},',
    f'Test CTCVR AUC={test_ctcvr_auc_value:.6f}',
  ])

  print(txt)
  