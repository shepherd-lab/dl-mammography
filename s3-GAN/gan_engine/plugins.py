import torch
import pandas as pd
import torch.nn.functional as F
from collections import OrderedDict
import pickle
import colors
from sklearn.metrics import roc_auc_score
import numpy as np
import os
from os.path import join as pjoin
from xz import debug, RunningAvg
import datetime
from tqdm import tqdm


class Checkpoint:

    def __init__(self, save_path_prefix='checkpoints/checkpoint', interval=1):
        self.save_path_prefix = save_path_prefix
        self.interval = interval

    def checkpoint(self, engine):
        save_path = pjoin(engine.pwd, f"{self.save_path_prefix}_{engine.i_epoch}.pth")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(engine.state_dict(), save_path)
        engine.epoch_end_events.append(f"Engine saved to {save_path}.")

    def begin(self, engine):
        self.i = 0

    def epoch_end(self, engine):
        self.i += 1
        if self.i >= self.interval:
            self.checkpoint(engine)
            self.i = 0


class Messages:

    def __init__(self, save_log_filename='log.txt', save_old_log_filename='old_log.txt'):
        self.save_log_filename = save_log_filename
        self.save_old_log_filename = save_old_log_filename

    def reset(self, engine):
        path_to_log = pjoin(engine.pwd, self.save_log_filename)
        path_to_old_log = pjoin(engine.pwd, self.save_old_log_filename)

        try:
            # if the log file already exists, move its content into the old log
            with open(path_to_log, 'r') as f, open(path_to_old_log, 'a') as f_old:
                f_old.write(f.read())
        except FileNotFoundError:
            pass

        # empty the current log file
        open(path_to_log, 'w').close()

    def begin(self, engine):
        self.pbar = tqdm(dynamic_ncols=True)

    def classifier_epoch_begin(self, engine):
        self.pbar.n = self.i_iter = 0
        self.pbar.total = len(engine.train_loader)
        self.pbar.set_description(
            ' | '.join(
                [
                    f"{engine.i_epoch}/{engine.max_epochs}",
                    colors.color(f"classifier", fg='black', bg='red'),
                    f"starting",
                ]
            )
        )

        engine.log = OrderedDict()
        engine.epoch_end_events = []

    def classifier_iter_begin(self, engine):
        engine.iter_end_display = []

    def classifier_iter_end(self, engine):
        self.i_iter += 1
        self.pbar.n = self.i_iter
        self.pbar.set_description(
            ' | '.join(
                [
                    f"{engine.i_epoch}/{engine.max_epochs}",
                    colors.color(f"classifier", fg='black', bg='red'),
                    f"{self.i_iter}/{len(engine.train_loader)}",
                    *engine.iter_end_display,
                ]
            )
        )

    def classifier_epoch_end(self, engine):
        self.pbar.n = self.pbar.total
        self.pbar.set_description(
            ' | '.join([
                f"{engine.i_epoch}/{engine.max_epochs}",
                colors.color(f"classifier", fg='black', bg='red'),
                f"done",
            ])
        )

    def detective_epoch_begin(self, engine):
        self.pbar.n = self.i_iter = 0
        self.pbar.total = len(engine.train_loader)
        self.pbar.set_description(
            ' | '.join(
                [
                    f"{engine.i_epoch}/{engine.max_epochs}",
                    colors.color(f"detective", fg='black', bg='red'),
                    f"starting",
                ]
            )
        )

        engine.log = OrderedDict()
        engine.epoch_end_events = []

    def detective_iter_begin(self, engine):
        engine.iter_end_display = []

    def detective_iter_end(self, engine):
        self.i_iter += 1
        self.pbar.n = self.i_iter
        self.pbar.set_description(
            ' | '.join(
                [
                    f"{engine.i_epoch}/{engine.max_epochs}",
                    colors.color(f"detective", fg='black', bg='red'),
                    f"{self.i_iter}/{len(engine.train_loader)}",
                    *engine.iter_end_display,
                ]
            )
        )

    def detective_epoch_end(self, engine):
        self.pbar.n = self.pbar.total
        self.pbar.set_description(
            ' | '.join([
                f"{engine.i_epoch}/{engine.max_epochs}",
                colors.color(f"detective", fg='black', bg='red'),
                f"done",
            ])
        )

    def valid_epoch_begin(self, engine):
        self.pbar.n = self.i_iter = 0
        self.pbar.total = len(engine.valid_loader)
        self.pbar.set_description(
            ' | '.join([
                f"{engine.i_epoch}/{engine.max_epochs}",
                colors.color(f"valid", fg='black', bg='blue'),
                f"starting",
            ])
        )

    def valid_iter_end(self, engine):
        self.i_iter += 1
        self.pbar.n = self.i_iter
        self.pbar.set_description(
            ' | '.join(
                [
                    f"{engine.i_epoch}/{engine.max_epochs}",
                    colors.color(f"valid", fg='black', bg='blue'),
                    f"{self.i_iter}/{len(engine.valid_loader)}",
                ]
            )
        )

    def valid_epoch_end(self, engine):
        self.pbar.n = self.pbar.total
        self.pbar.set_description(
            ' | '.join([
                f"{engine.i_epoch}/{engine.max_epochs}",
                colors.color(f"valid", fg='black', bg='blue'),
                f"done",
            ])
        )

    def epoch_end(self, engine):
        log_items = {
            'i_epoch': engine.i_epoch,
            **engine.log,
        }

        path_to_log = pjoin(engine.pwd, self.save_log_filename)
        self.pbar.write(str(log_items))
        with open(path_to_log, 'a') as f:
            f.write(repr(log_items))
            f.write('\n')


class Timestamp:

    def __init__(self, residual_factor=0.95):
        self.now = datetime.datetime.now()

    def epoch_end(self, engine):
        now = datetime.datetime.now()
        engine.log['timestamp'] = now.isoformat()
        self.now = now


def _default_display_fn(metric_results):
    return [f"{k} {v['values'][-1]:.2f} ({v['ravg'].running_avg:.2f})" for k, v in metric_results.items()]


class TrainingMetrics:

    def __init__(
        self,
        classifier_metrics,
        detective_metrics,
        display_fn=_default_display_fn,
        residual_factor=0.95,
        reset_running_avg_between_epochs=False,
    ):
        self.classifier_metrics = classifier_metrics
        self.detective_metrics = detective_metrics
        self.reset_running_avg_between_epochs = reset_running_avg_between_epochs
        self.residual_factor = residual_factor
        debug(f"residual_factor is set to be {self.residual_factor}")
        self.display_fn = display_fn

    def begin(self, engine):
        self.classifier_metric_results = {}
        for metric_name in self.classifier_metrics:
            self.classifier_metric_results[metric_name] = {
                'values': [],
                'ravg': RunningAvg(residual_factor=self.residual_factor),
            }

        self.detective_metric_results = {}
        for metric_name in self.detective_metrics:
            self.detective_metric_results[metric_name] = {
                'values': [],
                'ravg': RunningAvg(residual_factor=self.residual_factor),
            }

    def classifier_epoch_begin(self, engine):
        for metric_name in self.classifier_metrics:
            self.classifier_metric_results[metric_name]['values'] = []
            if self.reset_running_avg_between_epochs:
                self.classifier_metric_results[metric_name]['ravg'].reset()

    def classifier_iter_end(self, engine):
        y = engine.y
        yp = engine.yp

        for metric_name in self.classifier_metrics:
            metric_value = self.classifier_metrics[metric_name](yp, y)
            self.classifier_metric_results[metric_name]['values'].append(metric_value)
            self.classifier_metric_results[metric_name]['ravg'].step(metric_value)

        engine.iter_end_display += self.display_fn(self.classifier_metric_results)

    def classifier_epoch_end(self, engine):
        for metric_name in self.classifier_metrics:
            avg_value = np.mean(self.classifier_metric_results[metric_name]['values'])
            engine.log['classifier_trn_avg_' + metric_name] = avg_value

    def detective_epoch_begin(self, engine):
        for metric_name in self.detective_metrics:
            self.detective_metric_results[metric_name]['values'] = []
            if self.reset_running_avg_between_epochs:
                self.detective_metric_results[metric_name]['ravg'].reset()

    def detective_iter_end(self, engine):
        z = engine.z
        zp = engine.zp

        for metric_name in self.detective_metrics:
            metric_value = self.detective_metrics[metric_name](zp, z)
            self.detective_metric_results[metric_name]['values'].append(metric_value)
            self.detective_metric_results[metric_name]['ravg'].step(metric_value)

        engine.iter_end_display += self.display_fn(self.detective_metric_results)

    def detective_epoch_end(self, engine):
        for metric_name in self.detective_metrics:
            avg_value = np.mean(self.detective_metric_results[metric_name]['values'])
            engine.log['detective_trn_avg_' + metric_name] = avg_value


class ValidationMetrics:

    def __init__(self, classifier_metrics, detective_metrics):
        self.y_list = None
        self.yp_list = None
        self.classifier_metrics = classifier_metrics
        self.detective_metrics = detective_metrics

    def valid_epoch_begin(self, engine):
        self.y_list = []
        self.yp_list = []

        self.z_list = []
        self.zp_list = []

    def valid_iter_end(self, engine):
        self.y_list.append(engine.y)
        self.yp_list.append(engine.yp)

        self.z_list.append(engine.z)
        self.zp_list.append(engine.zp)

    def epoch_end(self, engine):
        y = torch.cat(self.y_list)
        yp = torch.cat(self.yp_list)

        for metric_name in self.classifier_metrics:
            engine.log['classifier_val_' + metric_name] = self.classifier_metrics[metric_name](yp, y)

        z = torch.cat(self.z_list)
        zp = torch.cat(self.zp_list)

        for metric_name in self.detective_metrics:
            engine.log['detective_val_' + metric_name] = self.detective_metrics[metric_name](yp, y)


class ReduceLROnPlateau:

    def __init__(self, factor=0.3, patience=2, epsilon=1e-7, reset_threshold=1e-7):
        self.scheduler = None
        self.factor = factor
        self.patience = patience
        self.epsilon = epsilon
        self.reset_threshold = reset_threshold

    def reset(self, engine):
        self.count = 0
        self.best_val_loss = np.inf

        for optimizer in [
            engine.cnn_optimizer,
            engine.classifier_optimizer,
            engine.detective_optimizer,
        ]:
            for g in optimizer.param_groups:
                g['original_lr'] = g['lr']

    def epoch_end(self, engine):
        # TODO: IMPORTANT: make the criterion a bit more formal
        val_loss = engine.log['detective_val_loss']

        if self.best_val_loss <= val_loss + self.epsilon:
            self.count += 1
            if self.count > self.patience:
                lrs = []
                for optimizer in [
                    engine.cnn_optimizer,
                    engine.classifier_optimizer,
                    engine.detective_optimizer,
                ]:
                    for g in optimizer.param_groups:
                        g['lr'] = g['lr'] * self.factor
                        lrs.append(f"{g['lr']:.1e}")
                        if g['lr'] < self.reset_threshold:
                            g['lr'] = g['original_lr']

                engine.epoch_end_events.append('lrs -> ' + ' '.join(lrs))

                self.count = 0
        else:
            self.best_val_loss = val_loss

        lrs = []
        for optimizer in [
            engine.cnn_optimizer,
            engine.classifier_optimizer,
            engine.detective_optimizer,
        ]:
            for g in optimizer.param_groups:
                lrs.append(g['lr'])

        engine.log['lr'] = lrs

        # Metrics + Events

        # There needs to be a mechanism to convert the *logged items* into their *display format*.

        # engine.load_log()

        # The different parts of the engine will need to be saved at different frequencies.
        # For example, the metrics need to be saved much more frequently than do the weights.
