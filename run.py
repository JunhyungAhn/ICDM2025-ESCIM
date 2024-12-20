import fire

from Ali_CCP.run import run as run_ccp
from Ali_Express.run import run as run_express

def main(dataset, country=None):
  if dataset=='Ali-CCP':
    run_ccp()
  elif dataset=='Ali-Express':
    run_express(country=country)
  else:
    raise Exception('Invalid Dataset')

if __name__ == '__main__':
  fire.Fire(main)