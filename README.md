# Tacotron 2 And WaveGlow v1.10 For PyTorch
# 한국어 수정중입니다.


## Quick Start Guide

To train your model using mixed precision with Tensor Cores or using FP32,
perform the following steps using the default parameters of the Tacrotron 2
and WaveGlow model on the [LJ Speech](https://keithito.com/LJ-Speech-Dataset/)
dataset.

1. Clone the repository.
   ```bash
   git clone https://github.com/NVIDIA/DeepLearningExamples.git
   cd DeepLearningExamples/PyTorch/SpeechSynthesis/Tacotron2
   ```

2. Download and preprocess the dataset.
Use the `./scripts/prepare_dataset.sh` download script to automatically
download and preprocess the training, validation and test datasets. To run
this script, issue:
   ```bash
   bash scripts/prepare_dataset.sh
   ```

   Data is downloaded to the `./LJSpeech-1.1` directory (on the host).  The
   `./LJSpeech-1.1` directory is mounted to the `/workspace/tacotron2/LJSpeech-1.1`
   location in the NGC container.

3. Build the Tacotron 2 and WaveGlow PyTorch NGC container.
   ```bash
   bash scripts/docker/build.sh
   ```

4. Start an interactive session in the NGC container to run training/inference.
After you build the container image, you can start an interactive CLI session with:

   ```bash
   bash scripts/docker/interactive.sh
   ```

   The `interactive.sh` script requires that the location on the dataset is specified.
   For example, `LJSpeech-1.1`. To preprocess the datasets for Tacotron 2 training, use
   the `./scripts/prepare_mels.sh` script:
   ```bash
   bash scripts/prepare_mels.sh
   ```

   The preprocessed mel-spectrograms are stored in the `./LJSpeech-1.1/mels` directory.

5. Start training.
To start Tacotron 2 training, run:
   ```bash
   bash scripts/train_tacotron2.sh
   ```

   To start WaveGlow training, run:
   ```bash
   bash scripts/train_waveglow.sh
   ```

6. Start validation/evaluation.
Ensure your loss values are comparable to those listed in the table in the
[Results](#results) section. For both models, the loss values are stored in the `./output/nvlog.json` log file.

   After you have trained the Tacotron 2 and WaveGlow models, you should get
   audio results similar to the
   samples in the `./audio` folder. For details about generating audio, see the
   [Inference process](#inference-process) section below.

   The training scripts automatically run the validation after each training
   epoch. The results from the validation are printed to the standard output
   (`stdout`) and saved to the log files.

7. Start inference.
After you have trained the Tacotron 2 and WaveGlow models, you can perform
inference using the respective checkpoints that are passed as `--tacotron2`
and `--waveglow` arguments. Tacotron2 and WaveGlow checkpoints can also be downloaded from NGC:

   https://ngc.nvidia.com/catalog/models/nvidia:tacotron2pyt_fp16/files?version=3

   https://ngc.nvidia.com/catalog/models/nvidia:waveglow256pyt_fp16/files?version=2

   To run inference issue:

   ```bash
   python inference.py --tacotron2 <Tacotron2_checkpoint> --waveglow <WaveGlow_checkpoint> --wn-channels 256 -o output/ -i phrases/phrase.txt --fp16
   ```

   The speech is generated from lines of text in the file that is passed with
   `-i` argument. The number of lines determines inference batch size. To run
   inference in mixed precision, use the `--fp16` flag. The output audio will
   be stored in the path specified by the `-o` argument.

   You can also run inference on CPU with TorchScript by adding flag --cpu:
   ```bash
   export CUDA_VISIBLE_DEVICES=
   ```
   ```bash
   python inference.py --tacotron2 <Tacotron2_checkpoint> --waveglow <WaveGlow_checkpoint> --wn-channels 256 --cpu -o output/ -i phrases/phrase.txt
   ```

## Advanced

The following sections provide greater details of the dataset, running
training and inference, and the training results.

### Scripts and sample code

The sample code for Tacotron 2 and WaveGlow has scripts specific to a
particular model, located in directories `./tacotron2` and `./waveglow`, as well as scripts common to both
models, located in the `./common` directory. The model-specific scripts are as follows:

* `<model_name>/model.py` - the model architecture, definition of forward and
inference functions
* `<model_name>/arg_parser.py` - argument parser for parameters specific to a
given model
* `<model_name>/data_function.py` - data loading functions
* `<model_name>/loss_function.py` - loss function for the model

The common scripts contain layer definitions common to both models
(`common/layers.py`), some utility scripts (`common/utils.py`) and scripts
for audio processing (`common/audio_processing.py` and `common/stft.py`). In
the root directory `./` of this repository, the `./run.py` script is used for
training while inference can be executed with the `./inference.py` script. The
scripts `./models.py`, `./data_functions.py` and `./loss_functions.py` call
the respective scripts in the `<model_name>` directory, depending on what
model is trained using the `run.py` script.

### Parameters

In this section, we list the most important hyperparameters and command-line arguments,
together with their default values that are used to train Tacotron 2 and
WaveGlow models.

#### Shared parameters

* `--epochs` - number of epochs (Tacotron 2: 1501, WaveGlow: 1001)
* `--learning-rate` - learning rate (Tacotron 2: 1e-3, WaveGlow: 1e-4)
* `--batch-size` - batch size (Tacotron 2 FP16/FP32: 104/48, WaveGlow FP16/FP32: 10/4)
* `--amp` - use mixed precision training
* `--cpu` - use CPU with TorchScript for inference

#### Shared audio/STFT parameters

* `--sampling-rate` - sampling rate in Hz of input and output audio (22050)
* `--filter-length` - (1024)
* `--hop-length` - hop length for FFT, i.e., sample stride between consecutive FFTs (256)
* `--win-length` - window size for FFT (1024)
* `--mel-fmin` - lowest frequency in Hz (0.0)
* `--mel-fmax` - highest frequency in Hz (8.000)

#### Tacotron 2 parameters

* `--anneal-steps` - epochs at which to anneal the learning rate (500 1000 1500)
* `--anneal-factor` - factor by which to anneal the learning rate (FP16/FP32: 0.3/0.1)

#### WaveGlow parameters

* `--segment-length` - segment length of input audio processed by the neural network (8000)
* `--wn-channels` - number of residual channels in the coupling layer networks (512)


### Command-line options

To see the full list of available options and their descriptions, use the `-h`
or `--help` command line option, for example:
```bash
python train.py --help
```

The following example output is printed when running the sample:

```bash
Batch: 7/260 epoch 0
:::NVLOGv0.2.2 Tacotron2_PyT 1560936205.667271376 (/workspace/tacotron2/dllogger/logger.py:251) train_iter_start: 7
:::NVLOGv0.2.2 Tacotron2_PyT 1560936207.209611416 (/workspace/tacotron2/dllogger/logger.py:251) train_iteration_loss: 5.415428161621094
:::NVLOGv0.2.2 Tacotron2_PyT 1560936208.705905914 (/workspace/tacotron2/dllogger/logger.py:251) train_iter_stop: 7
:::NVLOGv0.2.2 Tacotron2_PyT 1560936208.706479311 (/workspace/tacotron2/dllogger/logger.py:251) train_iter_items/sec: 8924.00136085362
:::NVLOGv0.2.2 Tacotron2_PyT 1560936208.706998110 (/workspace/tacotron2/dllogger/logger.py:251) iter_time: 3.0393316745758057
Batch: 8/260 epoch 0
:::NVLOGv0.2.2 Tacotron2_PyT 1560936208.711485624 (/workspace/tacotron2/dllogger/logger.py:251) train_iter_start: 8
:::NVLOGv0.2.2 Tacotron2_PyT 1560936210.236668825 (/workspace/tacotron2/dllogger/logger.py:251) train_iteration_loss: 5.516331672668457
```


### Getting the data

The Tacotron 2 and WaveGlow models were trained on the LJSpeech-1.1 dataset.
This repository contains the `./scripts/prepare_dataset.sh` script which will automatically download and extract the whole dataset. By default, data will be extracted to the `./LJSpeech-1.1` directory. The dataset directory contains a `README` file, a `wavs` directory with all audio samples, and a file `metadata.csv` that contains audio file names and the corresponding transcripts.

#### Dataset guidelines

The LJSpeech dataset has 13,100 clips that amount to about 24 hours of speech. Since the original dataset has all transcripts in the `metadata.csv` file, in this repository we provide file lists in the `./filelists` directory that determine training and validation subsets; `ljs_audio_text_train_filelist.txt` is a test set used as a training dataset and `ljs_audio_text_val_filelist.txt` is a test set used as a validation dataset.

#### Multi-dataset

To use datasets different than the default LJSpeech dataset:

1. Prepare a directory with all audio files and pass it to the `--dataset-path` command-line option.

2. Add two text files containing file lists: one for the training subset (`--training-files`) and one for the validation subset (`--validation files`).
The structure of the filelists should be as follows:
   ```bash
   `<audio file path>|<transcript>`
   ```

   The `<audio file path>` is the relative path to the path provided by the `--dataset-path` option.

### Training process

The Tacotron2 and WaveGlow models are trained separately and independently.
Both models obtain mel-spectrograms from short time Fourier transform (STFT)
during training. These mel-spectrograms are used for loss computation in case
of Tacotron 2 and as conditioning input to the network in case of WaveGlow.

The training loss is averaged over an entire training epoch, whereas the
validation loss is averaged over the validation dataset. Performance is
reported in total output mel-spectrograms per second for the Tacotron 2 model and
in total output samples per second for the WaveGlow model. Both measures are
recorded as `train_iter_items/sec` (after each iteration) and
`train_epoch_items/sec` (averaged over epoch) in the output log file `./output/nvlog.json`. The result is
averaged over an entire training epoch and summed over all GPUs that were
included in the training.

Even though the training script uses all available GPUs, you can change
this behavior by setting the `CUDA_VISIBLE_DEVICES` variable in your
environment or by setting the `NV_GPU` variable at the Docker container launch
([see section "GPU isolation"](https://github.com/NVIDIA/nvidia-docker/wiki/nvidia-docker#gpu-isolation)).

### Inference process

You can run inference using the `./inference.py` script. This script takes
text as input and runs Tacotron 2 and then WaveGlow inference to produce an
audio file. It requires  pre-trained checkpoints from Tacotron 2 and WaveGlow
models and input text as a text file, with one phrase per line.

To run inference, issue:
```bash
python inference.py --tacotron2 <Tacotron2_checkpoint> --waveglow <WaveGlow_checkpoint> --wn-channels 256 -o output/ --include-warmup -i phrases/phrase.txt --fp16
```
Here, `Tacotron2_checkpoint` and `WaveGlow_checkpoint` are pre-trained
checkpoints for the respective models, and `phrases/phrase.txt` contains input
phrases. The number of text lines determines the inference batch size. Audio
will be saved in the output folder. The audio files [audio_fp16](./audio/audio_fp16.wav)
and [audio_fp32](./audio/audio_fp32.wav) were generated using checkpoints from
mixed precision and FP32 training, respectively.

You can find all the available options by calling `python inference.py --help`.

You can also run inference on CPU with TorchScript by adding flag --cpu:
```bash
export CUDA_VISIBLE_DEVICES=
```
```bash
python inference.py --tacotron2 <Tacotron2_checkpoint> --waveglow <WaveGlow_checkpoint> --wn-channels 256 --cpu -o output/ -i phrases/phrase.txt
```

## Performance

### Benchmarking

The following section shows how to run benchmarks measuring the model
performance in training and inference mode.

#### Training performance benchmark

To benchmark the training performance on a specific batch size, run:

**Tacotron 2**

* For 1 GPU
	* FP16
        ```bash
        python train.py -m Tacotron2 -o <output_dir> -lr 1e-3 --epochs 10 -bs <batch_size> --weight-decay 1e-6 --grad-clip-thresh 1.0 --cudnn-enabled --log-file nvlog.json --load-mel-from-disk --training-files=filelists/ljs_mel_text_train_subset_2500_filelist.txt --validation-files=filelists/ljs_mel_text_val_filelist.txt --dataset-path <dataset-path> --amp
        ```
	* TF32 (or FP32 if TF32 not enabled)
        ```bash
        python train.py -m Tacotron2 -o <output_dir> -lr 1e-3 --epochs 10 -bs <batch_size> --weight-decay 1e-6 --grad-clip-thresh 1.0 --cudnn-enabled --log-file nvlog.json --load-mel-from-disk --training-files=filelists/ljs_mel_text_train_subset_2500_filelist.txt --validation-files=filelists/ljs_mel_text_val_filelist.txt --dataset-path <dataset-path>
        ```

* For multiple GPUs
	* FP16
        ```bash
        python -m multiproc train.py -m Tacotron2 -o <output_dir> -lr 1e-3 --epochs 10 -bs <batch_size> --weight-decay 1e-6 --grad-clip-thresh 1.0 --cudnn-enabled --log-file nvlog.json --load-mel-from-disk --training-files=filelists/ljs_mel_text_train_subset_2500_filelist.txt --validation-files=filelists/ljs_mel_text_val_filelist.txt --dataset-path <dataset-path> --amp
        ```
	* TF32 (or FP32 if TF32 not enabled)
        ```bash
        python -m multiproc train.py -m Tacotron2 -o <output_dir> -lr 1e-3 --epochs 10 -bs <batch_size> --weight-decay 1e-6 --grad-clip-thresh 1.0 --cudnn-enabled --log-file nvlog.json --load-mel-from-disk --training-files=filelists/ljs_mel_text_train_subset_2500_filelist.txt --validation-files=filelists/ljs_mel_text_val_filelist.txt --dataset-path <dataset-path>
        ```

**WaveGlow**
  
* For 1 GPU
	* FP16
        ```bash
        python train.py -m WaveGlow -o <output_dir> -lr 1e-4 --epochs 10 -bs <batch_size> --segment-length 8000 --weight-decay 0 --grad-clip-thresh 65504.0 --cudnn-enabled --cudnn-benchmark --log-file nvlog.json --training-files filelists/ljs_audio_text_train_subset_1250_filelist.txt --dataset-path <dataset-path> --amp
        ```
	* TF32 (or FP32 if TF32 not enabled)
        ```bash
        python train.py -m WaveGlow -o <output_dir> -lr 1e-4 --epochs 10 -bs <batch_size> --segment-length  8000 --weight-decay 0 --grad-clip-thresh 3.4028234663852886e+38 --cudnn-enabled --cudnn-benchmark --log-file nvlog.json --training-files filelists/ljs_audio_text_train_subset_1250_filelist.txt --dataset-path <dataset-path>
        ```

* For multiple GPUs
	* FP16
        ```bash
        python -m multiproc train.py -m WaveGlow -o <output_dir> -lr 1e-4 --epochs 10 -bs <batch_size> --segment-length 8000 --weight-decay 0 --grad-clip-thresh 65504.0 --cudnn-enabled --cudnn-benchmark --log-file nvlog.json --training-files filelists/ljs_audio_text_train_subset_1250_filelist.txt --dataset-path <dataset-path> --amp
        ```
	* TF32 (or FP32 if TF32 not enabled)
        ```bash
        python -m multiproc train.py -m WaveGlow -o <output_dir> -lr 1e-4 --epochs 10 -bs <batch_size> --segment-length 8000 --weight-decay 0 --grad-clip-thresh 3.4028234663852886e+38 --cudnn-enabled --cudnn-benchmark --log-file nvlog.json --training-files filelists/ljs_audio_text_train_subset_1250_filelist.txt --dataset-path <dataset-path>
        ```

Each of these scripts runs for 10 epochs and for each epoch measures the
average number of items per second. The performance results can be read from
the `nvlog.json` files produced by the commands.

#### Inference performance benchmark

To benchmark the inference performance on a batch size=1, run:

* For FP16
    ```bash
    python inference.py --tacotron2 <Tacotron2_checkpoint> --waveglow <WaveGlow_checkpoint> -o output/ --include-warmup -i phrases/phrase_1_64.txt --fp16 --log-file=output/nvlog_fp16.json
    ```
* For TF32 (or FP32 if TF32 not enabled)
    ```bash
    python inference.py --tacotron2 <Tacotron2_checkpoint> --waveglow <WaveGlow_checkpoint> -o output/ --include-warmup -i phrases/phrase_1_64.txt --log-file=output/nvlog_fp32.json
    ```

The output log files will contain performance numbers for Tacotron 2 model
(number of output mel-spectrograms per second, reported as `tacotron2_items_per_sec`)
and for WaveGlow (number of output samples per second, reported as `waveglow_items_per_sec`).
The `inference.py` script will run a few warmup iterations before running the benchmark.
