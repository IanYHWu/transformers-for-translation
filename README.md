# Transformers for Translation 
A PyTorch implementation of the transformer model for machine translation

This implementation includes:

- Support for both bilingual and multilingual translation
- Pivot model evaluation for zero-shot translation
- Support for distributed training
- BPE tokenisation
- Automatic BLEU score tracking
- Automatic interfacing with Wandb 

The default dataset is the TED-multi dataset from Huggingface https://huggingface.co/datasets/ted_multi. This can be easily changed by editing `preprocess.py`
to interface with other datasets. 
 

# Usage

Hyperparameters can be set in `hyperparams/config.yml`, and loaded via the argparse command `--custom_model`

To train a simple French to English model, run

`python train.py --name='model_name' --location='my_location' --langs en fr --custom_model='my_params'`

A folder named `model_name` will be automatically created in `my_location`, and this will contain a csv file documenting 
the results, a text file containing all the input parameters, a binary file containing the model checkpoint, and the
tokeniser used in preprocessing.

To train a multilingual German-English-French model, run

`python train.py --name='model_name' --location='my_location' --langs de en fr --custom_model='my_params'`

To evaluate a model, run

`python test.py --name='model_name' --location='my_location' --custom_model='my_params'`

This will automatically load the binary file contained in `my_location/model_name`. 
 
See `common/train_arguments.py` and `common/test_arguments.py` for more input options
