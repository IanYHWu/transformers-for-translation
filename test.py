"""
Inference Loop for MNMT
"""

import torch
import numpy as np
import time
from tokenizers import Tokenizer
from common.preprocess import detokenize, tokenize
from models import initialiser
from common import data_logger as logging
from common.metrics import BLEU
from common import preprocess
from common.test_arguments import TestParser
from hyperparams.loader import Loader
from common.utils import get_pairs, get_directions
from common.functional import inference_step


def test(device, params, test_dataloader, tokenizer, verbose=50):
    """Test loop"""

    logger = logging.TestLogger(params)
    logger.make_dirs()
    train_params = logging.load_params(params.location + '/' + params.name)

    model = initialiser.initialise_model(train_params, device)
    model, _, _, _ = logging.load_checkpoint(logger.checkpoint_path, device, model)

    test_batch_accs = []
    bleu = BLEU()
    bleu.set_excluded_indices([0, 2])

    test_acc = 0.0
    start_ = time.time()

    print("Now testing")
    for i, data in enumerate(test_dataloader):

        x, y = data
        test_batch_acc = inference_step(x, y, model, logger, tokenizer, device, bleu=bleu,
                                        teacher_forcing=params.teacher_forcing,
                                        beam_length=params.beam_length,
                                        alpha=params.alpha, beta=params.beta)
        test_batch_accs.append(test_batch_acc)

        test_acc += (test_batch_acc - test_acc) / (i + 1)
        curr_bleu = bleu.get_metric()

        if verbose is not None:
            if i % verbose == 0:
                print('Batch {} Accuracy {:.4f} Bleu {:.4f} in {:.4f} s per batch'.format(
                    i, test_acc, curr_bleu, (time.time() - start_) / (i + 1)))

    test_bleu = bleu.get_metric()
    direction = params.langs[0] + '-' + params.langs[1]
    logger.log_results([direction, test_acc, test_bleu])
    logger.dump_examples()


def multi_test(device, params, test_dataloader, tokenizer, verbose=50):
    """Test for multilingual translation. Evaluates on all possible translation directions."""

    logger = logging.TestLogger(params)
    logger.make_dirs()
    train_params = logging.load_params(params.location + '/' + params.name)

    model = initialiser.initialise_model(train_params, device)
    model, _, _, _ = logging.load_checkpoint(logger.checkpoint_path, device, model)

    assert tokenizer is not None
    add_targets = preprocess.AddTargetTokens(params.langs, tokenizer)
    pair_accs = {s + '-' + t: 0.0 for s, t in get_pairs(params.langs)}
    pair_bleus = {}
    for s, t in get_pairs(params.langs, excluded=params.excluded):
        _bleu = BLEU()
        _bleu.set_excluded_indices([0, 2])
        pair_bleus[s + '-' + t] = _bleu

    test_acc = 0.0
    start_ = time.time()

    print("Now testing")
    for i, data in enumerate(test_dataloader):

        data = get_directions(data, params.langs, excluded=params.excluded)
        for direction, (x, y, y_lang) in data.items():
            x = add_targets(x, y_lang)
            bleu = pair_bleus[direction]
            test_batch_acc = inference_step(x, y, model, logger, tokenizer, device, bleu=bleu,
                                            teacher_forcing=params.teacher_forcing,
                                            beam_length=params.beam_length)
            pair_accs[direction] += (test_batch_acc - pair_accs[direction]) / (i + 1)

        # report the mean accuracy and bleu accross directions
        if verbose is not None:
            test_acc += (np.mean([v for v in pair_accs.values()]) - test_acc) / (i + 1)
            curr_bleu = np.mean([bleu.get_metric() for bleu in pair_bleus.values()])
            if i % verbose == 0:
                print('Batch {} Accuracy {:.4f} Bleu {:.4f} in {:.4f} s per batch'.format(
                    i, test_acc, curr_bleu, (time.time() - start_) / (i + 1)))

    directions = [d for d in pair_bleus.keys()]
    test_accs = [pair_accs[d] for d in directions]
    test_bleus = [pair_bleus[d].get_metric() for d in directions]
    logger.log_results([directions, test_accs, test_bleus])
    logger.dump_examples()


def pivot_test(device, params, test_dataloader_1, test_dataloader_2, tokenizer_1, tokenizer_2, verbose=50):
    logger = logging.TestLogger(params)
    logger.make_dirs()
    train_params = logging.load_params(params.location + '/' + params.name)

    model_1 = initialiser.initialise_model(train_params, device)
    state_1 = torch.load(params.pivot_model_1, map_location=device)
    model_1.load_state_dict(state_1['model_state_dict'])

    model_2 = initialiser.initialise_model(train_params, device)
    state_2 = torch.load(params.pivot_model_2, map_location=device)
    model_2.load_state_dict(state_2['model_state_dict'])

    test_batch_accs = []
    bleu = BLEU()
    bleu.set_excluded_indices([0, 2])

    test_acc = 0.0
    start_ = time.time()

    for i, (data_1, data_2) in enumerate(zip(test_dataloader_1, test_dataloader_2)):
        x_1, y_1 = data_1
        x_2, y_2 = data_2

        y_pred_1 = inference_step(x_1, y_1, model_1, logger, tokenizer_1, device,
                                  teacher_forcing=params.teacher_forcing, pivot_mode=True)
        y_pred_det = detokenize(y_pred_1, tokenizer_1[1])
        y_pred_tok = tokenize(y_pred_det, tokenizer_2[0])
        test_batch_acc = inference_step(y_pred_tok, y_2, model_2, logger, tokenizer_2, device,
                                        teacher_forcing=params.teacher_forcing, pivot_mode=False, bleu=bleu)

        test_batch_accs.append(test_batch_acc)
        test_acc += (test_batch_acc - test_acc) / (i + 1)
        curr_bleu = bleu.get_metric()

        if verbose is not None:
            if i % verbose == 0:
                print('Batch {} Accuracy {:.4f} Bleu {:.4f} in {:.4f} s per batch'.format(
                    i, test_acc, curr_bleu, (time.time() - start_) / (i + 1)))

    test_bleu = bleu.get_metric()
    direction = "pivot"
    logger.log_results([direction, test_acc, test_bleu])
    logger.dump_examples()


def main(params):
    """ Loads the dataset and trains the model."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if len(params.langs) == 2:
        # bilingual translation
        # check for existing tokenizers
        if params.tokenizer is not None:
            tokenizers = [Tokenizer.from_file('pretrained/' + tok + '.json') for tok in params.tokenizer]
        else:
            try:
                tokenizers = [Tokenizer.from_file(params.location + '/' + lang + '_tokenizer.json') for lang in
                          params.langs]
            except:
                tokenizers = None

        train_dataloader, val_dataloader, test_dataloader, tokenizers = preprocess.load_and_preprocess(
            params.langs, params.batch_size, params.vocab_size, params.dataset,
            tokenizer=tokenizers, multi=False)

        test(device, params, test_dataloader, tokenizers, verbose=params.verbose)

    elif len(params.langs) > 2 and not params.pivot:
        # multilingual translation
        #  check for existing tokenizers
        if params.tokenizer is not None:
            tokenizer = Tokenizer.from_file('pretrained/' + params.tokenizer + '.json')
        else:
            try:
                tokenizer = Tokenizer.from_file(params.location + '/multi_tokenizer.json')
            except:
                tokenizer = None

        train_dataloader, val_dataloader, test_dataloader, tokenizer = preprocess.load_and_preprocess(
            params.langs, params.batch_size, params.vocab_size, params.dataset,
            tokenizer=tokenizer, multi=True)

        multi_test(device, params, test_dataloader, tokenizer, verbose=params.verbose)

    elif len(params.langs) > 2 and params.pivot:
        try:
            tokenizer_1_1 = Tokenizer.from_file(params.pivot_tokenizer_path_1_1)
            tokenizer_1_2 = Tokenizer.from_file(params.pivot_tokenizer_path_1_2)
            tokenizer_2_1 = Tokenizer.from_file(params.pivot_tokenizer_path_2_1)
            tokenizer_2_2 = Tokenizer.from_file(params.pivot_tokenizer_path_2_2)
            tokenizer_1 = [tokenizer_1_1, tokenizer_1_2]
            tokenizer_2 = [tokenizer_2_1, tokenizer_2_2]
        except:
            tokenizer_1 = None
            tokenizer_2 = None

        test_dataloader_1, test_dataloader_2 = preprocess.pivot_load_and_preprocess(params.langs,
                                                                                    params.batch_size,
                                                                                    params.dataset,
                                                                                    tokenizer_1=tokenizer_1,
                                                                                    tokenizer_2=tokenizer_2)

        pivot_test(device, params, test_dataloader_1, test_dataloader_2, tokenizer_1, tokenizer_2,
                   verbose=params.verbose)


if __name__ == "__main__":
    args = TestParser.parse_args()

    # Loader can also take in any dictionary of parameters
    params = Loader(args, check_custom=True)
    main(params)
