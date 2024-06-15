import datetime
import os
import tempfile
import torch
from sklearn.metrics import mean_squared_error, mean_absolute_error
from transformers import Trainer, TrainingArguments, set_seed, EarlyStoppingCallback
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from tsfm_public.toolkit.time_series_preprocessor import TimeSeriesPreprocessor, get_datasets, ScalerType, \
    StandardScaler
from tsfm_public.models.tinytimemixer import TinyTimeMixerForPrediction
from tsfm_public.models.tinytimemixer.utils import plot_preds, count_parameters
from tsfm_public.toolkit.callbacks import TrackingCallback


class TimeSeriesModelTrainer:
    def __init__(self, data_file, out_dir, model_revision, seed=42):
        self.data_file = data_file
        self.out_dir = out_dir
        self.model_revision = model_revision
        self.seed = seed
        set_seed(self.seed)
        self.df = self._load_data()
        self.tsp = self._init_preprocessor()
        # Move the model and datasets to GPU if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.predictions_zeroshot = None
        self.predictions_finetune = None

    def _load_data(self):
        df = pd.read_csv(self.data_file)
        df.index = pd.date_range("2022-06-01 00:00:01", freq="1s", periods=len(df))

        if df.isnull().values.any():
            print("Dataset contains missing values. Filling missing values using back and forward fill")
            df = df.ffill()
        return df

    def _init_preprocessor(self):
        timestamp_column = "Timestamp"
        id_columns = []
        target_columns = self.df.columns[1:].tolist()

        column_specifiers = {
            "timestamp_column": timestamp_column,
            "id_columns": id_columns,
            "target_columns": target_columns,
            "control_columns": [],
        }

        tsp = TimeSeriesPreprocessor(
            **column_specifiers,
            context_length=512,
            prediction_length=96,
            scaling=True,
            encode_categorical=False,
            scaler_type=ScalerType.STANDARD.value
        )
        return tsp

    def zeroshot_eval(self, batch_size, prediction_filter_length=None):
        zeroshot_model = TinyTimeMixerForPrediction.from_pretrained("ibm/TTM", revision=self.model_revision,
                                                                    prediction_filter_length=prediction_filter_length)
        temp_dir = tempfile.mkdtemp()

        zeroshot_trainer = Trainer(
            model=zeroshot_model.to(self.device),  # Move model to GPU
            args=TrainingArguments(
                output_dir=temp_dir,
                per_device_eval_batch_size=batch_size,
            )
        )

        _, _, test_dataset = get_datasets(ts_preprocessor=self.tsp, dataset=self.df, split_config=self._split_config(),
                                          fewshot_fraction=1.0)

        # Make predictions using the zero-shot model
        #zeroshot_output = zeroshot_trainer.predict(test_dataset)
        #self.predictions_zeroshot = zeroshot_output.predictions

        #print("+" * 20, "Validation MSE zero-shot", "+" * 20)
        #print(zeroshot_output.metrics)

        plot_preds(trainer=zeroshot_trainer, dset=test_dataset, plot_dir=os.path.join(self.out_dir, "plots"),
                   plot_prefix="test_zeroshot_test5", channel=6, num_plots=8)

    def fewshot_finetune_eval(self, batch_size, learning_rate=0.001, forecast_length=96,
                              fewshot_percent=5, freeze_backbone=True, num_epochs=10, save_dir=None,
                              prediction_filter_length=None):
        if save_dir is None:
            save_dir = self.out_dir

        out_dir = os.path.join(save_dir, "finetune")

        print("-" * 20, f"Running few-shot {fewshot_percent}%", "-" * 20)

        train_dataset, valid_dataset, test_dataset = get_datasets(ts_preprocessor=self.tsp,
                                                                  dataset=self.df,
                                                                  split_config=self._split_config(),
                                                                  fewshot_fraction=fewshot_percent / 100,
                                                                  fewshot_location="first")


        if prediction_filter_length is None:
            finetune_forecast_model = TinyTimeMixerForPrediction.from_pretrained("ibm/TTM",
                                                                                 revision=self.model_revision)
        elif prediction_filter_length <= forecast_length:
            finetune_forecast_model = TinyTimeMixerForPrediction.from_pretrained("ibm/TTM",
                                                                                 revision=self.model_revision,
                                                                                 prediction_filter_length=prediction_filter_length)
        else:
            raise ValueError(f"`prediction_filter_length` should be <= `forecast_length`")

        if freeze_backbone:
            print("Number of params before freezing backbone", count_parameters(finetune_forecast_model))
            for param in finetune_forecast_model.backbone.parameters():
                param.requires_grad = False
            print("Number of params after freezing the backbone", count_parameters(finetune_forecast_model))

        finetune_forecast_args = TrainingArguments(
            output_dir=os.path.join(out_dir, "output"),
            overwrite_output_dir=True,
            learning_rate=learning_rate,
            num_train_epochs=num_epochs,
            do_eval=True,
            eval_strategy="epoch",
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            dataloader_num_workers=4,
            report_to=None,
            save_strategy="epoch",
            logging_strategy="epoch",
            save_total_limit=1,
            logging_dir=os.path.join(out_dir, "logs"),
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,

        )

        early_stopping_callback = EarlyStoppingCallback(
            early_stopping_patience=5,
            early_stopping_threshold=0.001,
        )
        tracking_callback = TrackingCallback()

        optimizer = AdamW(finetune_forecast_model.parameters(), lr=learning_rate)
        scheduler = OneCycleLR(optimizer, learning_rate, epochs=num_epochs,
                               steps_per_epoch=math.ceil(len(train_dataset) / batch_size))

        finetune_forecast_trainer = Trainer(
            model=finetune_forecast_model,
            args=finetune_forecast_args,
            train_dataset=train_dataset,
            eval_dataset=valid_dataset,
            callbacks=[early_stopping_callback, tracking_callback],
            optimizers=(optimizer, scheduler),
        )

        finetune_forecast_trainer.train()

        # Make predictions using the fine-tuned model
        finetune_output = finetune_forecast_trainer.evaluate(test_dataset)
        #self.predictions_finetune = finetune_output.predictions

        print("+" * 20, f"Test MSE after few-shot {fewshot_percent}% fine-tuning", "+" * 20)
        print(finetune_output)
        print(finetune_output.metrics)

        plot_preds(trainer=finetune_forecast_trainer, dset=test_dataset, plot_dir=os.path.join(self.out_dir, "plots"),
                   plot_prefix="test_fewshot_better_plot", channel=-1)

    def _split_config(self):
        return {
            "train": [0, int(12 * 24 * 60 * 60 * 0.9)],
            "valid": [int(12 * 24 * 60 * 60 * 0.9), int(12 * 24 * 60 * 60 * 0.95)],
            "test": [int(12 * 24 * 60 * 60 * 0.95), int(12 * 24 * 60 * 60 * 1.0)]
        }


def data_preproccesing(data):
    df = pd.read_csv(data)
    df.index = pd.date_range("2022-06-01 00:00:01", freq="1s", periods=len(df))
    df = df.rename_axis('Timestamp')
    df.reset_index(inplace=True)
    return df


def inference(dataframe, context_length, prediction_length, start):
    # Ensure the 'Timestamp' column is in datetime format
    if not np.issubdtype(dataframe['Timestamp'].dtype, np.datetime64):
        dataframe['Timestamp'] = pd.to_datetime(dataframe['Timestamp'])

    # Extracting the past and future dataframes based on context and prediction lengths
    dataframe_past = dataframe.iloc[start:start + context_length]
    dataframe_future = dataframe.iloc[start + context_length:start + context_length + prediction_length]

    # Displaying the first few rows of the past and future dataframes
    print("Past Dataframe:")
    print(dataframe_past.head())
    print("Future Dataframe:")
    print(dataframe_future.head())

    # Printing the lengths of the past and future dataframes
    print("Past dataframe length: ", len(dataframe_past))
    print("Future dataframe length: ", len(dataframe_future))

    # Combining the past and future dataframes for plotting
    dataframe_combined = pd.concat([dataframe_past, dataframe_future], axis=0)
    print("Combined Dataframe Columns:", dataframe_combined.columns)

    print("Number of rows: ", dataframe.info())

    # Plotting each column in the dataframe
    for column in dataframe_combined.columns:
        if column == 'Timestamp':
            continue
        else:
            scaler = StandardScaler()
            scaled_column = scaler.fit_transform(dataframe_combined[[column]])
            plt.figure(figsize=(20, 6))

            # Adjusting timestamp for the vertical line
            last_timestamp = dataframe_combined['Timestamp'].iloc[-1]
            adjusted_timestamp = last_timestamp - datetime.timedelta(seconds=96)

            # Plotting the vertical line
            plt.axvline(x=adjusted_timestamp, color='red', label='Prediction Start')

            # Plotting the column values over time
            plt.plot(dataframe_combined['Timestamp'], scaled_column)

            plt.title(f'Time Series Plot for {column}')
            plt.xlabel('Timestamp')
            plt.ylabel(column)
            plt.legend()

            # Formatting the x-axis for better tick visibility
            ax = plt.gca()
            locator = mdates.SecondLocator(interval=30)  # Adjust the interval as needed
            formatter = mdates.DateFormatter('%Y-%m-%d %H:%M:%S')
            ax.xaxis.set_major_locator(locator)
            ax.xaxis.set_major_formatter(formatter)
            plt.xticks(rotation=45)

            plt.tight_layout()

    plt.show()


if __name__ == "__main__":
    trainer = TimeSeriesModelTrainer(data_file='vdg_dataset_column.csv', out_dir="ttm_finetuned_models/", model_revision="main")
    # Perform zero-shot evaluation
    #trainer.zeroshot_eval(batch_size=64)
    # Perform few-shot fine-tuning and evaluation
    #trainer.fewshot_finetune_eval(batch_size=64)
    #print(trainer.predictions_zeroshot)
    #print(trainer.predictions_finetune)
    df = data_preproccesing('vdg_dataset_column.csv')
    inference(df, 512, 96, 1026175)

