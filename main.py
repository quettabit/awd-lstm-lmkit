# sys packages
import gc
import math
import os
import sys
import time

# third party packages
from comet_ml import Experiment

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


# our packages
from utils.data import CorpusLoader
from models.model import Model
from utils.cli_parser import CLIParser 
from utils.main import create_exp_dir, save_checkpoint, save_recent, copy_assets
from utils.main import exp, calc_rank, emb_patch_and_freeze
from utils.main import parameter_counts, batchify, get_batch, parallelize_module
from utils.main import get_per_param_options, repackage_hidden
from utils.main import get_model_args, parse_model_result, get_optimizer
from utils.main import get_criterion_args, inspect_grad
from utils.main import init_random, continue_random, get_state
from utils.graceful_killer import GracefulKiller
from utils.logger import Logger
from utils.real_metrics import TopMetrics
from utils.loss_criterion import LossCriterion


def get_experiment_objects():
    args = CLIParser().parse_args()
    comet = Experiment(api_key="<your_key>", project_name="<your_project>", 
                        workspace="<your_workspace>", log_code=False, 
                        auto_param_logging=False, auto_metric_logging=False, 
                        disabled=args.no_comet, display_summary=False)

    if not args.continue_train:
        # else args.save is the directory from which assets are loaded to 
        # continue training
        args.save = "{}-{}".format(args.save, time.strftime("%Y%m%d-%H%M%S"))
        create_exp_dir(args)

    comet.set_name(args.save.split('/')[-1])
    comet.log_parameters(vars(args))
    copy_assets(args, comet)

    last_state = None
    if args.continue_train:
        last_state = torch.load(os.path.join(args.save, 'state.pt'))
        continue_random(last_state, args.cuda)
    else:
        init_random(args.seed, args.cuda)

    return args, comet, last_state


def get_data_objects(args):
    corpus = CorpusLoader.load(args.data)
    train_data = batchify(corpus.train, args.batch_size, args)
    val_data = batchify(corpus.valid, args.eval_batch_size, args)
    test_data = batchify(corpus.test, args.test_batch_size, args)
    return train_data, val_data, test_data, len(corpus.dictionary)


def get_learning_objects(args, last_state, lgr):
    model = Model(args)
    model = emb_patch_and_freeze(model, args)
    criterion = LossCriterion(args)
    if args.continue_train:
        model.load_state_dict(torch.load(os.path.join(args.save, 'model.pt')))
        start_epoch = last_state['epoch'] + 1
        # Tip !! if args.epoch is set to values less than -2,
        # no training is done, evaluation is done on test set and 
        # continues with analysis if analysis flag is set.
        lgr.log('-' * 89)
        lgr.log("Continuing training from epoch %s" % (start_epoch))
        lgr.log('-' * 89)
    else:
        start_epoch = 1
        lgr.log('Args: {}'.format(args))
        lgr.log(
            "Model total parameters: {}".format(
                parameter_counts(model, criterion)
            ),
            print_=True
        )
    args.start_epoch = start_epoch # patch into args
    model = parallelize_module(model, args)
    criterion = parallelize_module(criterion, args)
    optimizer = get_optimizer(model, criterion, args)
    return model, criterion, optimizer


def evaluate(model, data_source, batch_size, is_test=False):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    if args.rnn_type == 'QRNN':
        model.base_model.reset()
    total_raw_loss = 0
    hidden = model.base_model.init_hidden(batch_size)
    with torch.no_grad():
        for i in range(0, data_source.size(0) - 1, args.bptt):
            data, targets = get_batch(data_source, i, args)
            targets = targets.view(-1)
            # ensure output is 2D
            output, hidden = parse_model_result(
                model(**get_model_args(args, data, hidden, None, False)),
                top_metrics, False
            )
            loss, raw_loss = criterion(**get_criterion_args(
                args.criterion, output, targets, model.top_model
            ))

            total_raw_loss += raw_loss.data * len(data)

            hidden = repackage_hidden(hidden)
            if args.local_debug:
                return total_raw_loss.item()

    return total_raw_loss.item() / len(data_source)


def train(epoch, args, model, criterion, optimizer, 
             train_data, top_metrics, lgr):
    """
    returns a tuple of avg loss (raw loss, total loss)
    """
    assert args.batch_size % args.small_batch_size == 0, \
        'batch_size must be divisible by small_batch_size'
    with torch.autograd.set_detect_anomaly(args.detect_anomaly):
        if args.rnn_type == 'QRNN':
            model.base_model.reset()
        # Turn on training mode which enables dropout.
        total_raw_loss = 0
        total_cri_loss = 0
        total_loss = 0 # includes loss from regularization terms
        epoch_raw_loss = 0 # avg of raw loss across all batches in a single epoch
        epoch_loss = 0 
        logged_counter = 0

        start_time = time.time()
        if args.cuda and torch.cuda.is_available():
            lgr.log(
                "Current cuda memory usage is {} bytes".format(
                    torch.cuda.memory_allocated()
                )
            )
        hidden = [ model.base_model.init_hidden(args.small_batch_size) 
                    for _ in range(args.batch_size // args.small_batch_size) ]
        batch, i = 0, 0
        # train_data.size: 
        #   [77465, 12] for PTB with batchsize of 12
        #   [26107, 80] for WK2 with batchsize of 80
        #   [139241, 15] for WK2 with batchsize of 15
        while i < train_data.size(0) - 1 - 1:
            bptt = args.bptt if np.random.random() < 0.95 else args.bptt / 2.
            # Prevent excessively small or negative sequence lengths
            seq_len = max(5, int(np.random.normal(bptt, 5)))
            # There's a very small chance that it could select a very long 
            # sequence length resulting in OOM
            seq_len = min(seq_len, args.bptt + args.max_seq_len_delta)
            for param_group in optimizer.param_groups:
                param_group['prev_lr'] = param_group['lr']
                param_group['lr'] = param_group['lr'] * seq_len / args.bptt

            model.train()
            data, targets = get_batch(train_data, i, args, seq_len=seq_len)

            optimizer.zero_grad()

            start, end, s_id = 0, args.small_batch_size, 0
            while start < args.batch_size:
                cur_data, cur_targets = data[:, start: end], \
                                    targets[:, start: end].contiguous().view(-1)

                # Starting each batch, we detach the hidden state from how it 
                # was previously produced.
                # If we didn't, the model would try backpropagating all the way 
                # to start of the dataset.
                hidden[s_id] = repackage_hidden(hidden[s_id])
                # ensure output is 2D
                output, hidden[s_id],\
                    rnn_hs, dropped_rnn_hs = parse_model_result(
                        model(**get_model_args(
                            args, cur_data, hidden[s_id], cur_targets, True
                        )),
                        top_metrics, True
                    )
                
                loss, raw_loss = criterion(**get_criterion_args(
                    args.criterion, output, cur_targets, model.top_model
                ))
                loss_factor = args.small_batch_size / args.batch_size
                total_cri_loss += loss.data * loss_factor
                # Activiation Regularization
                loss = loss + sum(args.alpha * dropped_rnn_h.pow(2).mean() \
                        for dropped_rnn_h in dropped_rnn_hs[-1:])
                # Temporal Activation Regularization (slowness)
                loss = loss +\
                        sum(args.beta * (rnn_h[1:] - rnn_h[:-1]).pow(2).mean() \
                            for rnn_h in rnn_hs[-1:])
                loss *= loss_factor
                total_raw_loss += raw_loss.data * loss_factor
                total_loss += loss.data 
               
                loss.backward()

                s_id += 1
                start = end
                end = start + args.small_batch_size
    

                gc.collect()


                if args.local_debug:
                    break
            
            # `clip_grad_norm` helps prevent the exploding gradient problem in 
            # RNNs / LSTMs.
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            #inspect_grad(model.named_parameters())
            optimizer.step()

            # total_raw_loss += raw_loss.data

            for param_group in optimizer.param_groups:
                param_group['lr'] = param_group['prev_lr']

            log_now = False
            batch += 1
            i += seq_len

            if batch % args.log_interval == 0:
                log_now = True
                num_batches = args.log_interval
            elif i >= train_data.size(0):
                log_now = True
                num_batches = batch % args.log_interval
            """
            every time seq len is different and not exactly args.bptt
            so len(train_data) // args.bptt to count the number of batches 
            is not correct
            """
            if log_now:
                cur_raw_loss = total_raw_loss.item() / num_batches
                cur_cri_loss = total_cri_loss.item() / num_batches
                cur_loss = total_loss.item() / num_batches
                epoch_raw_loss += cur_raw_loss
                epoch_loss += cur_loss
                logged_counter += 1
                elapsed = time.time() - start_time
                lgr.log(
                    "| {} | epoch {:3d} | {:5d} batches done | lr {:02.4f} "
                    "| ms/batch {:5.2f} | raw loss {:5.2f} | raw ppl {:8.2f} "
                    "| cri. loss {:5.2f} | tot. loss {:5.2f}"
                    "".format(
                        time.strftime("%Y%m%d-%H%M%S"), epoch, batch, 
                        args.lr, 
                        elapsed * 1000 / num_batches, cur_raw_loss, 
                        exp(cur_raw_loss), cur_cri_loss, cur_loss
                    )
                )
                total_raw_loss = 0
                total_cri_loss = 0
                total_loss = 0
                start_time = time.time()

            if args.local_debug:
                logged_counter = 1
                break
       
        
        return (epoch_raw_loss/logged_counter, epoch_loss/logged_counter)

def learn(args, comet, killer, model, criterion, optimizer, train_data, 
            val_data, top_metrics, lgr):
    best_val_loss = []
    stored_loss = 100000000

    for epoch in range(args.start_epoch, args.epochs + args.start_epoch):
        epoch_start_time = time.time()
        avg_loss = train(epoch, args, model, criterion, optimizer, 
                            train_data, top_metrics, lgr)
        train_end_time = time.time()
        top_metrics.push('train', epoch)
        current_state = get_state(epoch)
        if args.save_recent:
            save_recent(model, optimizer, args.save, current_state)

        if 't0' in optimizer.param_groups[0]:
            tmp = {}
            for prm in model.parameters():
                tmp[prm] = prm.data.clone()
                if 'ax' in optimizer.state[prm]:
                    prm.data = optimizer.state[prm]['ax'].clone()

            val_loss = evaluate(model, val_data, args.eval_batch_size)

            if val_loss < stored_loss:
                save_checkpoint(model, optimizer, args.save, current_state)
                lgr.log('Saving Averaged!')
                stored_loss = val_loss

            for prm in model.parameters():
                if prm in tmp:
                    prm.data = tmp[prm].clone()

        else:
            val_loss = evaluate(model, val_data, args.eval_batch_size)

            if val_loss < stored_loss:
                save_checkpoint(model, optimizer, args.save, current_state)
                lgr.log('Saving Normal!')
                stored_loss = val_loss

            if not args.no_switch and args.optimizer == 'sgd':
                switch_needed = False
                if args.switch_epoch == -1 and\
                    't0' not in optimizer.param_groups[0] and\
                    (len(best_val_loss)>args.nonmono and\
                    val_loss > min(best_val_loss[:-args.nonmono])):
                        lgr.log('Non monotonically triggered!')
                        switch_needed = True
                elif args.switch_epoch == epoch:
                    lgr.log('Epoch triggered!')
                    switch_needed = True

                if switch_needed or args.local_debug:
                    lgr.log('Switching!')
                    optimizer = torch.optim.ASGD(
                        get_per_param_options(model, criterion, args), 
                        lr=args.lr, t0=0, lambd=0.
                    )
                    if args.local_debug:
                        print("switching optim in local debug mode!")

            elif args.optimizer == 'adam' and\
                    optimizer.param_groups[0]['lr'] == args.lr:
                switch_needed = False
                if args.switch_epoch == epoch:
                    lgr.log('Epoch triggered!')
                    switch_needed = True
                if switch_needed or args.local_debug:
                    lgr.log('Decreasing learning rate!')
                    for param_group in optimizer.param_groups:
                        param_group['lr'] /= 10
                    if args.local_debug:
                        print("decreasing lr in local debug mode!")

            best_val_loss.append(val_loss)

        top_metrics.push('val', epoch)
        comet.log_metrics(
            {
                'train_ppl': exp(avg_loss[0]),
                'valid_ppl': exp(val_loss)
            },
            step=epoch
        )

        lgr.log('-' * 89)
        lgr.log(
            "| {} | end of epoch {:3d} | train time: {:5.2f}s "
            "| total time: {:5.2f}s | avg. train raw loss {:5.2f} "
            "| avg. train raw ppl {:8.2f} | avg. train tot. loss {:5.2f} "
            "| valid loss {:5.2f} | valid ppl {:8.2f}"
            "".format(
                time.strftime("%Y%m%d-%H%M%S"), epoch, 
                (train_end_time - epoch_start_time),
                (time.time() - epoch_start_time), avg_loss[0], exp(avg_loss[0]),
                avg_loss[1], val_loss, exp(val_loss)
            )
        )
        lgr.log('-' * 89)
        

        if args.local_debug:
            print("epoch %s in debug mode done!" % (epoch))
            if epoch == args.start_epoch + 1:
                break
        if killer.has_kill_request():
            lgr.log('Killer has got a kill request.')
            break
    
    killer.notify_completion()

def test(args, comet, model, test_data, top_metrics, lgr, recent_model=False):
    # Run on test data.
    test_loss = evaluate(model, test_data, args.test_batch_size)
    if not recent_model:
        log_msg = 'End of training'
        comet_metric = 'test_ppl'
    else:
        log_msg = 'Recent model on test set'
        comet_metric = 'recent_model_test_ppl'
    top_metrics.push('test', 0)
    lgr.log('=' * 89)
    lgr.log(
        "| {} | {} | test loss {:5.2f} | test ppl {:8.2f}"
        "".format(
            time.strftime("%Y%m%d-%H%M%S"), log_msg, test_loss, exp(test_loss)
        )
    )
    lgr.log('=' * 89)
    comet.log_metric(comet_metric, exp(test_loss))

def analysis(args, comet, model, val_data, test_data, lgr, recent_model=False):
    # Post training analysis
    # Calculate ranks
    if not args.local_debug and not args.no_analysis:
        lgr.log('=' * 89)
        if not recent_model:
            lgr.log('For the best performing model,')
        else:
            lgr.log('For the recent model,')
        if 'penn' in args.data:
            # rank on validation only for PTB
            # due to OOM issues and time consumption, restricting only to 
            # test set for wk2
            
            val_ranks = calc_rank(
                model, val_data, args.eval_batch_size, args, 'val', comet
            )
            lgr.log(
                "| {} | Rank analysis on val data | {} "
                "".format(
                    time.strftime("%Y%m%d-%H%M%S"), str(val_ranks)
                )
            )
            comet.log_metrics(val_ranks, prefix='val')

        test_ranks = calc_rank(
            model, test_data, args.test_batch_size, args, 'test', comet
        )
        lgr.log(
            "| {} | Rank analysis on test data | {} "
            "".format(
                time.strftime("%Y%m%d-%H%M%S"), str(test_ranks)
            )
        )
        comet.log_metrics(test_ranks, prefix='test')
        lgr.log("| {} | End of analysis ".format(time.strftime("%Y%m%d-%H%M%S")))
        lgr.log('=' * 89)


if __name__ == '__main__':
    print('experiment started.')
    args, comet, last_state = get_experiment_objects()
    lgr = Logger(args.save)
    top_metrics = TopMetrics(comet)
    train_data, val_data, test_data, args.ntoken = get_data_objects(args)
    model, criterion, optimizer = get_learning_objects(args, last_state, lgr)
    
    
    killer = GracefulKiller()
    while not killer.kill_now:
        learn(args, comet, killer, model, criterion, optimizer, train_data, 
                val_data, top_metrics, lgr)

    # Load the best saved model.
    model.load_state_dict(torch.load(os.path.join(args.save, 'model.pt')))
    model = parallelize_module(model, args)

    test(args, comet, model, test_data, top_metrics, lgr)
    analysis(args, comet, model, val_data, test_data, lgr)
    if args.recent_model_analysis:
        model.load_state_dict(
            torch.load(os.path.join(args.save, 'model_recent.pt'))
        )
        model = parallelize_module(model, args)
        test(args, comet, model, test_data, top_metrics, lgr, recent_model=True)
        analysis(args, comet, model, val_data, test_data, lgr, recent_model=True)
    print('experiment done.')
