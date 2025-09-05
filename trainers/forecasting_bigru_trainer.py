import time
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import List

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from preprocessings import sliding, splitting
from algorithms.optimizers_cythonize import AdaptiveMomentEstimation
from modelings.forecasting_bigru_model_cythonize import ForecastingBiGRUModel

from warnings import filterwarnings
filterwarnings("ignore")


"""

  -- Forecasting BiGRU Trainer -- 

  Writer : Al-Fariqy Raihan Azhwar
  NPM    : 202143501514
  Class  : R8Q
  Email  : alfariqyraihan@gmail.com

"""


class ForecastingBiGRUTrainer:
  def __init__(
    self, 
    csv_file:           str,
    model_registry:     str,
    sequence_length:    int = 60, 
    hidden_sizes: List[int] = [256, 128, 64],
    batch_size:         int = 20, 
    epochs:             int = 25,
    learning_rate:    float = 0.001,
    patience:           int = 15
  ) -> None:
    self.torch_device:  torch.device = self.__get_torch_device()
    self.csv_file:               str = csv_file
    self.model_registry:         str = model_registry

    self.sequence_length:        int = sequence_length
    self.hidden_sizes:     List[int] = hidden_sizes
    self.batch_size:             int = batch_size
    self.epochs:                 int = epochs
    self.learning_rate:        float = learning_rate
    self.patience:               int = patience

    self.__load_data()
    self.__init_model()
    self.__init_training_components()


  """
    [ name ]:
      __get_torch_device (return dtype: torch.device)

    [ description ]
      get torch device
  """
  def __get_torch_device(self) -> torch.device:
    if torch.cuda.is_available():
      return torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
      return torch.device('mps')
    else:
      return torch.device('cpu')


  """
    [ name ]:
      __load_data (return dtype: None)

    [ description ]
      load data -> sliding, splitting
  """
  def __load_data(self) -> None:
    dataframe: pd.DataFrame = pd.read_csv(
      filepath_or_buffer = f'indonesia_stocks/modeling_datas/{self.csv_file}', 
      index_col          = 'Date'
    )
    
    dataframe.index = pd.to_datetime(dataframe.index)
    dataframe.dropna(inplace = True)

    sequences, labels = sliding(dataframe, seq_len = self.sequence_length)
    x_train, x_test, y_train, y_test = splitting(sequences, labels)

    self.X_train: torch.tensor = torch.tensor(data = x_train, dtype = torch.float32) \
      .to(device = self.torch_device)
    self.y_train: torch.tensor = torch.tensor(data = y_train, dtype = torch.float32) \
      .reshape(-1, 1).to(device = self.torch_device)

    self.X_test: torch.tensor = torch.tensor(data = x_test, dtype = torch.float32) \
      .to(device = self.torch_device)
    self.y_test: torch.tensor = torch.tensor(data = y_test, dtype = torch.float32) \
      .reshape(-1, 1).to(device = self.torch_device)


  """
    [ name ]:
      __init_model (return dtype: None)

    [ description ]
      init model
  """
  def __init_model(self) -> None:
    input_size: int = self.X_train.shape[2]
    self.model: ForecastingBiGRUModel = ForecastingBiGRUModel(
      input_size          = input_size,
      hidden_sizes        = self.hidden_sizes,
      output_size         = 1,
      dropout_probability = 0.25
    ).to(self.torch_device)

    print(f"Input size: {input_size}")
    print(self.model)

    total_params: int = sum(
      parameter.numel() for parameter in self.model.parameters() \
        if parameter.requires_grad
    )
    print(f"Total Parameters: {total_params}")


  """
    [ name ]:
      __init_training_components (return dtype: None)

    [ description ]
      init training components
  """
  def __init_training_components(self) -> None:
    self.criterion: nn.MSELoss = nn.MSELoss()
    self.optimizer: AdaptiveMomentEstimation = \
      AdaptiveMomentEstimation(
        parameters    = self.model.parameters(),
        learning_rate = self.learning_rate
      )

    self.train_loader = DataLoader(
      dataset    = TensorDataset(self.X_train, self.y_train),
      batch_size = self.batch_size,
      shuffle    = True
    )
    self.train_losses: List[float] = []
    self.val_losses:   List[float] = []


  """
    [ name ]:
      train (return dtype: None)

    [ description ]
      training
  """
  def train(self) -> None:
    counter:         int = 0
    start:         float = time.time()
    best_val_loss: float = float('inf')

    for epoch in range(self.epochs):
      self.model.train()
      epoch_loss:    int = 0
      progress_bar: tqdm = tqdm(
        self.train_loader, 
        desc = f'Epoch {epoch+1}/{self.epochs}', unit = 'batch'
      )

      for batch_X, batch_y in progress_bar:
        preds: torch.Tensor = self.model(batch_X)
        loss:  torch.Tensor = self.criterion(preds, batch_y)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        epoch_loss += loss.item()
        progress_bar.set_postfix(batch_loss = loss.item())

      avg_loss: float = epoch_loss / len(self.train_loader)
      self.train_losses.append(avg_loss)

      val_loss: float = self.evaluate()
      self.val_losses.append(val_loss)

      improved: bool = val_loss < best_val_loss
      print(f"Train Loss: {avg_loss:.6f} - Val Loss: {val_loss:.6f}")
      print(f"Epoch {epoch+1}: val_loss {'improved' if improved else 'did not improve'} from {best_val_loss:.6f} to {val_loss:.6f}")

      if improved:
        best_val_loss: float = val_loss
        counter:         int = 0
        torch.save(self.model.state_dict(), 'best_model.pth')
        print("Saving model to best_model.pth\n")
      else:
        counter += 1
        if counter >= self.patience:
          print(f"Early stopping at epoch {epoch+1}")
          break

      duration: float = (time.time() - start) / 60
      print(f'Training Time (Exponential): {duration:.2f} minutes\n')


  """
    [ name ]:
      evaluate (return dtype: float)

    [ description ]
      evaluate
  """
  def evaluate(self) -> float:
    self.model.eval()
    with torch.no_grad():
      preds: torch.Tensor = self.model(self.X_test)
      return self.criterion(preds, self.y_test).item()


  def metrics_evaluation(self) -> None:
    self.model.eval()
    with torch.no_grad():
        y_pred: torch.Tensor = self.model(self.X_test).cpu().numpy()[:, -1]

    y_test: np.ndarray = np.array(self.y_test)
    y_pred: np.ndarray = np.array(y_pred)

    mae:  float = mean_absolute_error(y_test, y_pred)
    rmse: float = np.sqrt(mean_squared_error(y_test, y_pred))
    r2:   float = r2_score(y_test, y_pred)

    epsilon: float = 1e-8
    mape:    float = np.mean(
      np.abs(
        (y_test - y_pred) / (np.abs(y_test) + epsilon)
      )
    ) * 100

    print(f'MAE  : {mae:.9f}')
    print(f'MAPE : {mape:.9f}%')
    print(f'RMSE : {rmse:.9f}')
    print(f'R²   : {r2:.9f}')


  """
    [ name ]:
      export_to_onnx (return dtype: None)

    [ parameters ]
      - onnx_location (dtype: str)

    [ description ]
      export pth to onnx
  """
  def export_to_onnx(self) -> None:
    stock_name:    str = self.csv_file
    onnx_location: str = f'{self.model_registry}/{stock_name[:len(stock_name) - 4]}.onnx'
    model: ForecastingBiGRUModel = ForecastingBiGRUModel(
      input_size          = self.X_train.shape[2],
      hidden_sizes        = self.hidden_sizes,
      output_size         = 1,
      dropout_probability = 0.25,
      action              = 'export'
    )
    model.load_state_dict(state_dict = torch.load("best_model.pth"))
    model.eval()

    dummy_input: torch.Tensor = torch.randn(
      self.batch_size, self.sequence_length, 
      self.X_train.shape[2]
    )

    torch.onnx.export(
      model, dummy_input, onnx_location,
      input_names   = ['input'],
      output_names  = ['output'],
      opset_version = 15,
      dynamic_axes  = {
        'input':  {0: 'batch_size'},
        'output': {0: 'batch_size'}
      }
    )
