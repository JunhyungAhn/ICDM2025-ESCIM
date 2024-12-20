import gc
import torch
import torch.nn as nn

from torcheval.metrics import BinaryAUROC, BinaryPrecisionRecallCurve

from sklearn.metrics import auc
  
def eval_on_testset(model, device, test_loader):
  
  test_ctr_auc, test_cvr_auc, test_ctcvr_auc = BinaryAUROC().to('cpu'), BinaryAUROC().to('cpu'), BinaryAUROC().to('cpu')
  
  test_ctr_prauc, test_cvr_prauc, test_ctcvr_prauc = BinaryPrecisionRecallCurve().to('cpu'), BinaryPrecisionRecallCurve().to('cpu'), BinaryPrecisionRecallCurve().to('cpu')
  
  test_ctr_loss_fn, test_cvr_loss_fn, test_ctcvr_loss_fn = nn.BCELoss(), nn.BCELoss(), nn.BCELoss()
  
  test_ctr_loss, test_cvr_loss, test_ctcvr_loss = 0, 0, 0
  num_samples = 0
  
  model.eval()
  
  test_ctr_auc.reset()
  test_cvr_auc.reset()
  test_ctcvr_auc.reset()
  
  test_ctr_prauc.reset()
  test_cvr_prauc.reset()
  test_ctcvr_prauc.reset()
  
  for i, (categorical_fields, numerical_fields, labels) in enumerate(test_loader):
    categorical_fields, numerical_fields, labels = categorical_fields.to(device), numerical_fields.to(device), labels.to(device)
    click, conversion = labels[:, 0], labels[:, 1]
    
    click = torch.unsqueeze(click.to(device), 1).to(torch.float)
    conversion = torch.unsqueeze(conversion.to(device), 1).to(torch.float)
    
    click_idx = (click==1).to(torch.float32).nonzero()[:, 0]
    
    with torch.no_grad():
      output = model(categorical_fields, numerical_fields)
    pctr, pcvr, pctcvr = output['pctr'], output['pcvr'], output['pctcvr']
    
    pcvr_clicked = pcvr[click_idx]
    conversion_clicked = conversion[click_idx]
    
    pctr = pctr.squeeze().detach().cpu()
    click = click.squeeze().detach().cpu()
    pcvr_clicked = pcvr_clicked.squeeze().detach().cpu()
    conversion_clicked = conversion_clicked.squeeze().detach().cpu()
    pctcvr = pctcvr.squeeze().detach().cpu()
    conversion = conversion.squeeze().detach().cpu()
    
    # AUC
    test_ctr_auc.update(pctr, click)
    test_cvr_auc.update(pcvr_clicked, conversion_clicked)
    test_ctcvr_auc.update(pctcvr, conversion)
    
    # PRAUC
    test_ctr_prauc.update(pctr, click)
    test_cvr_prauc.update(pcvr_clicked, conversion_clicked)
    test_ctcvr_prauc.update(pctcvr, conversion)
    
    # Log Loss
    # Compute loss for this batch
    ctr_loss = test_ctr_loss_fn(pctr, click)
    cvr_loss = test_cvr_loss_fn(pcvr_clicked, conversion_clicked)
    ctcvr_loss = test_ctcvr_loss_fn(pctcvr, conversion)
    
    # Accumulate total loss and number of samples
    test_ctr_loss += ctr_loss.item() * click.size(0)  # Multiply loss by batch size
    test_cvr_loss += cvr_loss.item() * conversion_clicked.size(0)  # Multiply loss by batch size
    test_ctcvr_loss += ctcvr_loss.item() * click.size(0)  # Multiply loss by batch size
    
    num_samples += click.size(0)
    
    del categorical_fields, numerical_fields, labels, click, conversion, output, pctr, pcvr, pctcvr, pcvr_clicked, conversion_clicked

    if i%100==0:
      gc.collect()
      torch.cuda.empty_cache()

  # AUC
  test_ctr_auc_value = test_ctr_auc.compute().to("cpu").item()
  test_cvr_auc_value = test_cvr_auc.compute().to("cpu").item()
  test_ctcvr_auc_value = test_ctcvr_auc.compute().to("cpu").item()
  
  # PRAUC  
  test_ctr_precision, test_ctr_recall, _ = test_ctr_prauc.compute()
  test_cvr_precision, test_cvr_recall, _ = test_cvr_prauc.compute()
  test_ctcvr_precision, test_ctcvr_recall, _ = test_ctcvr_prauc.compute()
  
  test_ctr_prauc_value = auc(sorted(test_ctr_recall.numpy()), sorted(test_ctr_precision.numpy()))
  test_cvr_prauc_value = auc(sorted(test_cvr_recall.numpy()), sorted(test_cvr_precision.numpy()))
  test_ctcvr_prauc_value = auc(sorted(test_ctcvr_recall.numpy()), sorted(test_ctcvr_precision.numpy()))
  
  # Log Loss
  test_ctr_loss /= num_samples
  test_cvr_loss /= num_samples
  test_ctcvr_loss /= num_samples
  
  txt = '\n'.join([
    f'Test CTR AUC={test_ctr_auc_value:.6f},',
    f'Test CVR AUC={test_cvr_auc_value:.6f},',
    f'Test CTCVR AUC={test_ctcvr_auc_value:.6f}',
    
    f'Test CTR PRAUC={test_ctr_prauc_value:.6f},',
    f'Test CVR PRAUC={test_cvr_prauc_value:.6f},',
    f'Test CTCVR PRAUC={test_ctcvr_prauc_value:.6f}',
    
    f'Test CTR Log-Loss={test_ctr_loss:.6f},',
    f'Test CVR Log-Loss={test_cvr_loss:.6f},',
    f'Test CTCVR Log-Loss={test_ctcvr_loss:.6f}',
  ])

  print(txt)
  
  return test_ctr_auc_value, test_cvr_auc_value, test_ctcvr_auc_value, test_ctr_prauc_value, test_cvr_prauc_value, test_ctcvr_prauc_value
