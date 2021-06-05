"""
Training Loop for MNMT
"""
import torch
import torch.nn as nn
import torch.multiprocessing as mp
import torch.distributed as dist

import numpy as np
import os, sys, time
import wandb
from tokenizers import Tokenizer

import models.base_transformer as base_transformer
import models.initialiser as initialiser
from common import preprocess
from common.train_arguments import TrainParser
from common import data_logger as logging
from hyperparams.loader import Loader
from hyperparams.schedule import WarmupDecay
from common.metrics import BLEU
from common.utils import to_devices, sample_direction
from common.functional import train_step, val_step, beam_search

SEED = 1337


def seed_all(SEED):
    """ Set the seed for all devices. """
    torch.cuda.manual_seed_all(SEED)
    np.random.seed(SEED)


def setup(params):
    """ Create directories required and create logger. If checkpoint then
    some parameters are overwritten by command line arguments."""
    RESERVED = ['wandb', 'add_epochs', 'checkpoint', 'location', 'name']

    new_root_path = params.location
    new_name = params.name
    if params.checkpoint:
        prev_params = logging.load_params(new_root_path + '/' + new_name)
        for param, val in prev_params.__dict__.items():
            if param not in RESERVED:
                setattr(params, param, val)
        params.epochs += params.add_epochs
        logger = logging.TrainLogger(params)
        logger.make_dirs()
    else:
        logger = logging.TrainLogger(params)
        logger.make_dirs()
    logger.save_params()
    return logger, params


def train(rank, device, logger, params, train_dataloader, val_dataloader=None, tokenizer=None,
          verbose=50):
    """Training Loop"""

    multi = False
    if len(params.langs) > 2:
        assert tokenizer is not None
        multi = True
        add_targets = preprocess.AddTargetTokens(params.langs, tokenizer)

    model = initialiser.initialise_model(params, device)
    optimizer = torch.optim.Adam(model.parameters())
    scheduler = WarmupDecay(optimizer, params.warmup_steps, params.d_model, lr_scale=params.lr_scale)
    criterion = torch.nn.CrossEntropyLoss(reduction='none')

    _target = torch.tensor(1.0).to(device)

    epoch = 0
    if params.checkpoint:
        model, optimizer, epoch, scheduler = logging.load_checkpoint(logger.checkpoint_path, device, model,
                                                                     optimizer=optimizer, scheduler=scheduler)

    if params.distributed:
        model = nn.parallel.DistributedDataParallel(model, device_ids=[device.index],
                                                    find_unused_parameters=True)

    if rank == 0:
        if params.wandb:
            wandb.watch(model)
        batch_losses, batch_accs = [], []
        epoch_losses, epoch_accs = [], []
        val_epoch_losses, val_epoch_accs, val_epoch_bleus = [], [], []

    while epoch < params.epochs:
        start_ = time.time()

        # train
        if params.FLAGS:
            print('Training')
        epoch_loss = 0.0
        epoch_acc = 0.0
        for i, data in enumerate(train_dataloader):

            if multi:
                # sample a tranlsation direction and add target tokens
                (x, y), (x_lang, y_lang) = sample_direction(data, params.langs, excluded=params.excluded)
                x = add_targets(x, y_lang)
            else:
                x, y = data

            batch_loss, batch_acc = train_step(x, y, model, criterion, optimizer,
                                                          scheduler, device, distributed=params.distributed)

            if rank == 0:
                batch_loss = batch_loss.item()
                batch_acc = batch_acc.item()
                batch_losses.append(batch_loss)
                batch_accs.append(batch_acc)
                epoch_loss += (batch_loss - epoch_loss) / (i + 1)
                epoch_acc += (batch_acc - epoch_acc) / (i + 1)

                if verbose is not None:
                    if i % verbose == 0:
                        print('Batch {} Loss {:.4f} Accuracy {:.4f} in {:.4f} s per batch'.format(
                            i, epoch_loss, epoch_acc, (time.time() - start_) / (i + 1)))
                if params.wandb:
                    wandb.log({'loss': batch_loss, 'accuracy': batch_acc})

        if rank == 0:
            epoch_losses.append(epoch_loss)
            epoch_accs.append(epoch_acc)

        # val only on rank 0
        if rank == 0:
            if params.FLAGS:
                print('Validating')
            val_epoch_loss = 0.0
            val_epoch_acc = 0.0
            val_bleu = 0.0
            test_bleu = 0.0
            if val_dataloader is not None:
                bleu = BLEU()
                bleu.set_excluded_indices([0, 2])
                for i, data in enumerate(val_dataloader):
                    if multi:
                        # sample a tranlsation direction and add target tokens
                        (x, y), (x_lang, y_lang) = sample_direction(data, params.langs, excluded=params.excluded)
                        x = add_targets(x, y_lang)
                    else:
                        x, y = data

                    batch_loss, batch_acc = val_step(x, y, model, criterion, bleu, device,
                                                     distributed=params.distributed)

                    batch_loss = batch_loss.item()
                    batch_acc = batch_acc.item()
                    val_epoch_loss += (batch_loss - val_epoch_loss) / (i + 1)
                    val_epoch_acc += (batch_acc - val_epoch_acc) / (i + 1)

                val_epoch_losses.append(val_epoch_loss)
                val_epoch_accs.append(val_epoch_acc)
                val_bleu = bleu.get_metric()

                # evaluate without teacher forcing
                if params.test_freq is not None:
                    if epoch % params.test_freq == 0:
                        bleu_no_tf = BLEU()
                        bleu_no_tf.set_excluded_indices([0, 2])
                        for i, data in enumerate(val_dataloader):
                            if i > params.test_batches:
                                break
                            else:
                                if multi:
                                    # sample a tranlsation direction and add target tokens
                                    (x, y), (x_lang, y_lang) = sample_direction(data, params.langs,
                                                                                excluded=params.excluded)
                                    x = add_targets(x, y_lang)
                                else:
                                    x, y = data

                                y, y_tar = y[:, 0].unsqueeze(-1), y[:, 1:]
                                enc_mask, look_ahead_mask, dec_mask = base_transformer.create_masks(x, y_tar)

                                # devices
                                x, y, y_tar, enc_mask = to_devices((x, y, y_tar, enc_mask), device)

                                y_pred = beam_search(x, y, y_tar, model, enc_mask=enc_mask,
                                                     beam_length=params.beam_length, alpha=params.alpha,
                                                     beta=params.beta)
                                bleu_no_tf(y_pred, y_tar)

                        test_bleu = bleu_no_tf.get_metric()
                        print(test_bleu)

                if verbose is not None:
                    print('Epoch {} Loss {:.4f} Accuracy {:.4f} Val Loss {:.4f} Val Accuracy {:.4f} Val Bleu {:.4f}'
                          ' Test Bleu {:.4f} in {:.4f} secs \n'.format(epoch, epoch_loss, epoch_acc, val_epoch_loss,
                                                                       val_epoch_acc, val_bleu, test_bleu,
                                                                       time.time() - start_))
                if params.wandb:
                    wandb.log({'loss': epoch_loss, 'accuracy': epoch_acc, 'val_loss': val_epoch_loss,
                               'val_accuracy': val_epoch_acc, 'val_bleu': val_bleu, 'test_bleu': test_bleu})
            else:
                if verbose is not None:
                    print('Epoch {} Loss {:.4f} Accuracy {:.4f} in {:.4f} secs \n'.format(
                        epoch, epoch_loss, epoch_loss, epoch_acc, time.time() - start_))
                if params.wandb:
                    wandb.log({'loss': epoch_loss, 'accuracy': epoch_acc})

            if params.FLAGS:
                print('logging results')
            logger.save_model(epoch, model, optimizer, scheduler=scheduler)
            logger.log_results([epoch_loss, epoch_acc, val_epoch_loss, val_epoch_acc, val_bleu, test_bleu])

        epoch += 1

    return epoch_losses, epoch_accs, val_epoch_losses, val_epoch_accs


def main(gpu, params):
    """ Loads the dataset and trains the model."""
    rank = params.nr * params.gpus + gpu
    if params.distributed:
        dist.init_process_group(backend='nccl', init_method='env://',
                                world_size=params.world_size, rank=rank)
    seed_all(SEED)

    # get gpu device
    if params.device == 'gpu':
        device = torch.device(gpu)
    else:
        device = 'cpu'

    # only wandb on main process
    if rank == 0 and params.wandb:
        wandb.init(project='mnmt', entity='nlp-mnmt-project',
                   config={k: v for k, v in params.__dict__.items() if isinstance(v, (float, int, str))})
        config = wandb.config
    logger, params = setup(params)

    # load data and train for required experiment
    if len(params.langs) == 2:
        # bilingual translation

        #  load tokenizers if continuing
        if params.checkpoint:
            tokenizers = []
            for lang in params.langs:
                tokenizers.append(Tokenizer.from_file(logger.root_path + '/' + lang + '_tokenizer.json'))
        else:
            if params.tokenizer is not None:
                if len(params.tokenizer) == 2:
                    tokenizers = [Tokenizer.from_file('pretrained/' + tok + '.json') for tok in params.tokenizer]
                else:
                    print('Wrong number of tokenizers passed. Retraining.')
                    tokenizers = None
            else:
                tokenizers = None

        train_dataloader, val_dataloader, test_dataloader, _ = preprocess.load_and_preprocess(
            params.langs, params.batch_size, params.vocab_size, params.dataset, multi=False, path=logger.root_path,
            tokenizer=tokenizers, distributed=params.distributed, world_size=params.world_size, rank=rank)

        train(rank, device, logger, params, train_dataloader, val_dataloader=val_dataloader, verbose=params.verbose)

    elif len(params.langs) > 2:
        # multilingual translation

        #  load tokenizers if continuing
        if params.checkpoint:
            tokenizer = Tokenizer.from_file(logger.root_path + '/multi_tokenizer.json')
        else:
            if params.tokenizer is not None:
                tokenizer = Tokenizer.from_file('pretrained/' + params.tokenizer + '.json')
            else:
                tokenizer = None

        train_dataloader, val_dataloader, test_dataloader, tokenizer = preprocess.load_and_preprocess(
            params.langs, params.batch_size, params.vocab_size, params.dataset, multi=True, path=logger.root_path,
            tokenizer=tokenizer, distributed=params.distributed, world_size=params.world_size, rank=rank)

        train(rank, device, logger, params, train_dataloader, val_dataloader=val_dataloader, tokenizer=tokenizer,
              verbose=params.verbose)

    else:
        raise NotImplementedError

    # end wanb process to avoid hanging
    if rank == 0 and params.wandb:
        wandb.finish()


def run_distributed(params):
    params.world_size = params.gpus * params.nodes
    try:
        os.environ['MASTER_ADDR']
        os.environ['MASTER_PORT']
    except KeyError:
        print('Missing environment variable.')
        sys.exit(1)
    mp.spawn(main, nprocs=params.gpus, args=(params,))


if __name__ == "__main__":

    args = TrainParser.parse_args()
    # Loader can also take in any dictionary of parameters
    params = Loader(args, check_custom=True)
    if params.distributed:
        run_distributed(params)
    else:
        params.world_size = params.gpus * params.nodes
        main(0, params)
