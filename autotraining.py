import os
import gc
import json
from typing import List
from argparse import ArgumentParser, Namespace

from trainers.forecasting_bigru_trainer import ForecastingBiGRUTrainer


"""

  -- autotraining -- 

  Writer : Al-Fariqy Raihan Azhwar
  NPM    : 202143501514
  Class  : R8Q
  Email  : alfariqyraihan@gmail.com

"""


def main() -> None:
  try:
    parser: ArgumentParser = ArgumentParser(description = "thesis_forecasting_autotraining")

    parser.add_argument(
      '-w', '--workloads_json',
      type = str, required = True, help = 'workloads'
    )

    parser.add_argument(
      '-r', '--model_registry',
      type = str, required = True, help = 'workloads'
    )

    arguments:    Namespace = parser.parse_args()
    workloads_location: str = f'indonesia_stocks/workloads/{arguments.workloads_json}'

    with open(workloads_location, 'r') as json_file:
      workloads: List[str] = json.load(json_file).get('workloads', [])

    for workload in workloads:
      print('----------------------------------------------')
      print(f'------------------ {workload} --------------------')
      print('----------------------------------------------')
      trainer: ForecastingBiGRUTrainer = \
        ForecastingBiGRUTrainer(
          csv_file        = workload,
          model_registry  = arguments.model_registry,

          sequence_length = 60,
          # sequence_length = 14,
          hidden_sizes      = [256, 128, 64],
          # hidden_sizes    = [7],
          epochs = 25,
          # epochs = 3
        )
        
      trainer.train()
      trainer.metrics_evaluation()
      trainer.export_to_onnx()
      os.remove(path = "best_model.pth")

      del trainer; gc.collect()
    
  except Exception as error_message:
    print(error_message)

if __name__ == "__main__": main()
