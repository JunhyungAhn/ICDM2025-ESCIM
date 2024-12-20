import gc
import torch

from torcheval.metrics import BinaryAUROC
  
def eval_on_testset(model, device, test_loader):
  eval_metrics = {}
  
  test_ctr_auc, test_cvr_auc, test_ctcvr_auc = BinaryAUROC().to('cpu'), BinaryAUROC().to('cpu'), BinaryAUROC().to('cpu')
  
  model.eval()
  for i, (click, conversion, features) in enumerate(test_loader):
    for key in features.keys():
      features[key] = features[key].to(device)
    click = torch.unsqueeze(click.to(device), 1).to(torch.float)
    conversion = torch.unsqueeze(conversion.to(device), 1).to(torch.float)
    
    click_idx = (click==1).to(torch.float32).nonzero()[:, 0]
    
    with torch.no_grad():
      output = model(features)
    pctr, pcvr, pctcvr = output['pctr'], output['pcvr'], output['pctcvr']
    
    pcvr_clicked = pcvr[click_idx]
    conversion_clicked = conversion[click_idx]
    
    pctr = pctr.squeeze().detach().cpu()
    click = click.squeeze().detach().cpu()
    pcvr_clicked = pcvr_clicked.squeeze().detach().cpu()
    conversion_clicked = conversion_clicked.squeeze().detach().cpu()
    pctcvr = pctcvr.squeeze().detach().cpu()
    conversion = conversion.squeeze().detach().cpu()
    
    test_ctr_auc.update(pctr, click)
    test_cvr_auc.update(pcvr_clicked, conversion_clicked)
    test_ctcvr_auc.update(pctcvr, conversion)
    
    del features, key, click, conversion, output, pctr, pcvr, pctcvr, pcvr_clicked, conversion_clicked

    if i%100==0:
      gc.collect()
      torch.cuda.empty_cache()

  test_ctr_auc_value = test_ctr_auc.compute().to("cpu").item()
  test_cvr_auc_value = test_cvr_auc.compute().to("cpu").item()
  test_ctcvr_auc_value = test_ctcvr_auc.compute().to("cpu").item()
  
  eval_metrics['auc'] = {
    'ctr': test_ctr_auc_value,
    'cvr': test_cvr_auc_value,
    'ctcvr': test_ctcvr_auc_value,
  }
  
  txt = '\n'.join([
    f'Test CTR AUC={test_ctr_auc_value:.6f},',
    f'Test CVR AUC={test_cvr_auc_value:.6f},',
    f'Test CTCVR AUC={test_ctcvr_auc_value:.6f}',
  ])

  print(txt)
  
  return test_ctr_auc_value, test_cvr_auc_value, test_ctcvr_auc_value
