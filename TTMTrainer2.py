import os
import math
import tempfile
import torch
import numpy as np
import pandas as pd
from tsfm_public.models.tinytimemixer.utils import plot_preds, count_parameters
from tsfm_public.toolkit.callbacks import TrackingCallback
from tsfm_public.toolkit.time_series_preprocessor import TimeSeriesPreprocessor, ScalerType
from tsfm_public.models.tinytimemixer import TinyTimeMixerForPrediction
from tsfm_public.models.tinytimemixer.utils import plot_preds
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from transformers import EarlyStoppingCallback, Trainer, TrainingArguments, set_seed

# Set seed for reproducibility
SEED = 42
set_seed(SEED)

# Data loading, cleaning, manipulation, visualization, and saving tasks are performed in another notebook and saved in a github repository.
target_dataset = "vdg_dataset_column.csv"

# Results dir
OUT_DIR = "ttm_finetuned_models/"

# TTM model branch
TTM_MODEL_REVISION = "main"
TTM_INPUT_SEQ_LEN = 512


# Define data loaders using TSP from the tsfm library
def get_data(
        dataset_name: str,
        context_length,
        forecast_length,
        fewshot_fraction=1.0
):
    print(dataset_name, context_length, forecast_length)

    config_map = {
        "vdg_dataset_column.csv": {
            # "dataset_path": "https://raw.githubusercontent.com/matteorinalduzzi/TTM/main/datasets/venice/venice_small.csv",
            "dataset_path": "vdg_dataset_column.csv",
            "timestamp_column": "Timestamp",
            "id_columns": [],
            "target_columns": ["VDG_VIBRACIJE_VDG_LEVI_[MM_S]", 'VDG_STRUJA_VSV_52_L_[A]'],
            # Adjusted based on your actual dataset columns
            "control_columns": [],
            "split_config": {
                "train": 0.9,
                "test": 0.05,
            },
        },
    }
    if dataset_name not in config_map.keys():
        raise ValueError(
            f"Currently `get_data()` function supports the following datasets: {config_map.keys()}\n \
                         For other datasets, please provide the proper configs to the TimeSeriesPreprocessor (TSP) module."
        )

    dataset_path = config_map[dataset_name]["dataset_path"]
    timestamp_column = config_map[dataset_name]["timestamp_column"]
    id_columns = config_map[dataset_name]["id_columns"]
    target_columns = config_map[dataset_name]["target_columns"]
    split_config = config_map[dataset_name]["split_config"]
    control_columns = config_map[dataset_name]["control_columns"]

    if not target_columns:
        df_tmp_ = pd.read_csv(dataset_path)
        target_columns = list(df_tmp_.columns)
        target_columns.remove(timestamp_column)

    # Load data
    data = pd.read_csv(dataset_path)

    # Example of renaming and reindexing for time series
    output_column_names = [
        'VDG_VIBRACIJE_VDG_LEVI_[MM_S]',
        'VDG_TEMP_NAMOTAJA_MOT_VDG_L_[°C]'
    ]
    #data = data[output_column_names[0]]
    data.index = pd.date_range("2022-06-01 00:00:01", freq="1s", periods=len(data))
    data = data.rename_axis('Timestamp').reset_index()

    if data.isnull().values.any():
        print("Dataset contains missing values. Filling missing values using back and forward fill")
        data = data.ffill()

    column_specifiers = {
        "timestamp_column": timestamp_column,
        "id_columns": id_columns,
        "target_columns": target_columns,
        "control_columns": control_columns,

    }

    tsp = TimeSeriesPreprocessor(
        **column_specifiers,
        context_length=context_length,
        prediction_length=forecast_length,
        scaling=True,
        encode_categorical=False,
        scaler_type=ScalerType.STANDARD.value,
    )

    # Adjusted target_columns here to match your dataset
    train_dataset, valid_dataset, test_dataset = tsp.get_datasets(
        data, split_config, fewshot_fraction=fewshot_fraction, fewshot_location="first"
    )
    print(f"Data lengths: train = {len(train_dataset)}, val = {len(valid_dataset)}, test = {len(test_dataset)}")

    return train_dataset, valid_dataset, test_dataset


def zeroshot_eval(
        dataset_name,
        batch_size,
        context_length=TTM_INPUT_SEQ_LEN,
        forecast_length=96,
        prediction_filter_length=None
):
    # Get data
    _, _, dset_test = get_data(dataset_name=dataset_name,
                               context_length=context_length,
                               forecast_length=forecast_length,
                               fewshot_fraction=1.0
                               )

    # Load model
    if prediction_filter_length is None:
        zeroshot_model = TinyTimeMixerForPrediction.from_pretrained(
            "ibm/TTM", revision=TTM_MODEL_REVISION
        )
    else:
        if prediction_filter_length <= forecast_length:
            zeroshot_model = TinyTimeMixerForPrediction.from_pretrained(
                "ibm/TTM", revision=TTM_MODEL_REVISION, prediction_filter_length=prediction_filter_length
            )
        else:
            raise ValueError(f"`prediction_filter_length` should be <= `forecast_length")
    temp_dir = tempfile.mkdtemp()
    # zeroshot_trainer
    zeroshot_trainer = Trainer(
        model=zeroshot_model,
        args=TrainingArguments(
            output_dir=temp_dir,
            per_device_eval_batch_size=batch_size
        )
    )
    # evaluate = zero-shot performance
    print("+" * 20, "Test MSE zero-shot", "+" * 20)
    zeroshot_output = zeroshot_trainer.evaluate(dset_test)
    print(zeroshot_output)

    # plot
    plot_preds(trainer=zeroshot_trainer, dset=dset_test, plot_dir=os.path.join(OUT_DIR, dataset_name),
               plot_prefix="test_zeroshot", channel=0)


def fewshot_finetune_eval(
        dataset_name,
        batch_size=64,
        learning_rate=5e-5,
        context_length=TTM_INPUT_SEQ_LEN,
        forecast_length=96,
        fewshot_percent=10,
        freeze_backbone=True,
        num_epochs=30,
        save_dir=OUT_DIR,
        prediction_filter_length=None
):
    out_dir = os.path.join(save_dir, dataset_name)

    print("-" * 20, f"Running few-shot {fewshot_percent}%", "-" * 20)

    # Data prep: Get dataset
    dset_train, dset_val, dset_test = get_data(
        dataset_name,
        context_length,
        forecast_length,
        fewshot_fraction=fewshot_percent / 100
    )

    if prediction_filter_length is None:
        finetune_forecast_model = TinyTimeMixerForPrediction.from_pretrained(
            "ibm/TTM", revision=TTM_MODEL_REVISION,
        )
    elif prediction_filter_length <= forecast_length:
        finetune_forecast_model = TinyTimeMixerForPrediction.from_pretrained(
            "ibm/TTM", revision=TTM_MODEL_REVISION, prediction_filter_length=prediction_filter_length
        )
    else:
        raise ValueError(f"`prediction_filter_length` should be <= `forecast_length")

    if freeze_backbone:
        print(
            "Number of params before freezing backbone",
            count_parameters(finetune_forecast_model),
        )

        # Freeze the backbone of the model
        for param in finetune_forecast_model.backbone.parameters():
            param.requires_grad = False

        # Count params
        print(
            "Number of params after freezing the backbone",
            count_parameters(finetune_forecast_model),
        )

    print(f"Using learning rate = {learning_rate}")
    finetune_forecast_args = TrainingArguments(
        output_dir=os.path.join(out_dir, "output"),
        overwrite_output_dir=True,
        learning_rate=learning_rate,
        num_train_epochs=num_epochs,
        do_eval=True,
        eval_strategy="epoch",
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        dataloader_num_workers=8,
        report_to=None,
        save_strategy="epoch",
        logging_strategy="epoch",
        save_total_limit=1,
        logging_dir=os.path.join(out_dir, "logs"),  # Make sure to specify a logging directory
        load_best_model_at_end=True,  # Load the best model when training ends
        metric_for_best_model="eval_loss",  # Metric to monitor for early stopping
        greater_is_better=False,  # For loss
    )

    # Create the early stopping callback
    early_stopping_callback = EarlyStoppingCallback(
        early_stopping_patience=15,  # Number of epochs with no improvement after which to stop
        early_stopping_threshold=0.0,  # Minimum improvement required to consider as improvement
    )
    tracking_callback = TrackingCallback()

    # Optimizer and scheduler
    optimizer = AdamW(finetune_forecast_model.parameters(), lr=learning_rate)
    scheduler = OneCycleLR(optimizer, learning_rate, epochs=num_epochs,
                           steps_per_epoch=math.ceil(len(dset_train) / batch_size))

    finetune_forecast_trainer = Trainer(
        model=finetune_forecast_model,
        args=finetune_forecast_args,
        train_dataset=dset_train,
        eval_dataset=dset_val,
        callbacks=[early_stopping_callback, tracking_callback],
        optimizers=(optimizer, scheduler),
    )

    # Fine tune
    finetune_forecast_trainer.train()

    # Evaluation
    print("+" * 20, f"Test MSE after few-shot {fewshot_percent}% fine-tuning", "+" * 20)
    fewshot_output = finetune_forecast_trainer.evaluate(dset_val)
    print(fewshot_output)
    print("+" * 60)

    # plot
    plot_preds(trainer=finetune_forecast_trainer, dset=dset_val, plot_dir=os.path.join(OUT_DIR, dataset_name),
               plot_prefix="test_fewshot_vibracija", channel=0)
    # plot
    plot_preds(trainer=finetune_forecast_trainer, dset=dset_val, plot_dir=os.path.join(OUT_DIR, dataset_name),
               plot_prefix="test_fewshot_struja", channel=1)


# Example usage
#zeroshot_eval(dataset_name=target_dataset, batch_size=64)
fewshot_finetune_eval(dataset_name=target_dataset)