import torch
import matplotlib.pyplot as plt
import pandas as pd
from xz import info, debug, warn
import numpy as np
from tqdm import tqdm
from scipy.stats import pearsonr

# TODO: add resuming functionality


def predict(engine, data_loader):
    yp_list = []
    engine.model.eval()
    with torch.no_grad():
        for i_iter, batch in enumerate(data_loader):
            x, y = engine.process_batch_fn(batch)
            yp = engine.model(x)
            yp_list.append(yp)

    return yp_list


def _default_process_batch_fn(batch, device):
    id_, x_, y_, z_ = batch
    x = x_.to(device=device)
    y = y_.to(device=device)
    z = z_.to(device=device)
    return id_, x, y, z


class Engine:

    def __init__(
        self,
        cnn_model,
        cnn_optimizer,
        classifier_model,
        classifier_optimizer,
        classifier_loss_fn,
        detective_model,
        detective_optimizer,
        detective_loss_fn,
        max_epochs,
        loader_pair,
        plugins,
        device,
        pwd,
        process_batch_fn=_default_process_batch_fn,
    ):
        self.cnn_model = cnn_model
        self.cnn_optimizer = cnn_optimizer[0](cnn_model.parameters(), **cnn_optimizer[1])

        self.classifier_model = classifier_model
        self.classifier_optimizer = classifier_optimizer[0](classifier_model.parameters(), **classifier_optimizer[1])
        self.classifier_loss_fn = classifier_loss_fn

        self.detective_model = detective_model
        self.detective_optimizer = detective_optimizer[0](detective_model.parameters(), **detective_optimizer[1])
        self.detective_loss_fn = detective_loss_fn

        self.max_epochs = max_epochs
        self.train_loader, self.valid_loader = loader_pair
        self.plugins = plugins
        self.device = device
        self.pwd = pwd
        self.process_batch_fn = process_batch_fn
        self.reset()

        self.cnn_model.to(device=self.device)
        self.classifier_model.to(device=self.device)
        self.detective_model.to(device=self.device)

    def state_dict(self):
        return {
            'cnn_model': self.cnn_model.state_dict(),
            'cnn_optimizer': self.cnn_optimizer.state_dict(),
            'classifier_model': self.classifier_model.state_dict(),
            'classifier_optimizer': self.classifier_optimizer.state_dict(),
            'detective_model': self.detective_model.state_dict(),
            'detective_optimizer': self.detective_optimizer.state_dict(),
            'sess': self.sess,
        }

    def load_state_dict(self, state_dict):
        self.cnn_model.load_state_dict(state_dict['cnn_model'])
        self.cnn_model.to(device=self.device)

        self.cnn_optimizer.load_state_dict(state_dict['cnn_optimizer'])
        self.cnn_optimizer.to(device=self.device)

        self.classifier_model.load_state_dict(state_dict['classifier_model'])
        self.classifier_model.to(device=self.device)

        # self.classifier_optimizer.load_state_dict(state_dict['classifier_optimizer'])
        # self.classifier_optimizer.to(device=self.device)

        self.detective_model.load_state_dict(state_dict['detective_model'])
        self.detective_model.to(device=self.device)

        # self.detective_optimizer.load_state_dict(state_dict['detective_optimizer'])
        # self.detective_optimizer.to(device=self.device)

        self.sess = state_dict['sess']

    def run_hooks(self, hook_name):
        for plugin in self.plugins:
            try:
                # the engine itself is sent to all the plugin hooks
                method = getattr(plugin, hook_name)
            except AttributeError:
                # hook does not exist in this plugin, pass
                pass
            else:
                method(self)

    def reset(self):
        self.i_epoch = 0
        self.sess = {}
        self.run_hooks('reset')

    # TODO: Peter: turn the density into classes instead of a real value
    # TODO: Try freezing the main network and keep training the adversial to confirm if the code is right, but keep in mind it could overfit.
    def obtain_penultimate_layer_vecs_from_train(self, test_run=False):
        """This function is written to obtain the penultimate layer vectors
        from the training set. Is is for isolated training.
        """
        # self.cnn_model.eval()
        # self.classifier_model.eval()

        # with torch.no_grad():
        #     pbar = tqdm(total=len(self.train_loader))
        #     df_list = []
        #     for i_iter, batch in enumerate(self.train_loader):
        #         if test_run and i_iter >= 3:
        #             break
        #         ids, x, y, z = self.process_batch_fn(batch, self.device)
        #         pen_vec = self.cnn_model(x)
        #         yp = self.classifier_model(pen_vec)
        #         zp = self.detective_model(pen_vec)

        #         # because these outputs are in batches, we need to pick each one of them out
        #         # before appending to the df_list
        #         ys = y.cpu().numpy()
        #         zs = z.cpu().numpy()
        #         pen_vecs = pen_vec.cpu().detach().numpy()
        #         yps = yp.cpu().detach().numpy()
        #         zps = zp.cpu().detach().numpy()

        #         for i in range(len(ids)):
        #             pbar.set_description(str(ids[i]))
        #             df_list.append(
        #                 {
        #                     'id_': ids[i],
        #                     'cancer_label': ys[i],
        #                     'density_label': zs[i],
        #                     'pen_vec': pen_vec[i],
        #                     'cancer_prediction': yps[i],
        #                     'density_prediction': zps[i],
        #                 }
        #             )

        #         pbar.update()
        #     pbar.close()

        # torch.save(df_list, 'tmp/pen_vecs.pth')
        df_list = torch.load('tmp/pen_vecs.pth')

        df = pd.DataFrame.from_records(
            {
                'id_': [x['id_'] for x in df_list],
                'cancer_prediction': [x['cancer_prediction'][0] for x in df_list],
                'density_label': [x['density_label'][0] for x in df_list],
            }
        )
        print(f"rows with NA:")
        print(df.loc[df.isna().any(axis=1)])

        df = df.dropna()
        print(f"corr(cancer_prediction, density_label) = {pearsonr(df['cancer_prediction'], df['density_label'])}")

    def alternate_train_and_validate(self, test_run=False):
        if test_run:
            warn('WARNING: this is a test run')

        self.run_hooks('begin')

        while self.i_epoch < self.max_epochs:
            self.run_hooks('epoch_begin')

            self.cnn_model.train()
            self.classifier_model.train()
            self.detective_model.train()

            #
            # train the classifier:
            #
            #     · freeze the detector
            #     · loss = classifier_loss - detective_loss
            #

            # self.run_hooks('classifier_epoch_begin')
            # for i_iter, batch in enumerate(self.train_loader):
            #     if test_run and i_iter >= 3:
            #         break
            #     self.run_hooks('classifier_iter_begin')
            #     _, self.x, self.y, _ = _, x, y, _ = self.process_batch_fn(batch, self.device)
            #     self.cnn_optimizer.zero_grad()
            #     self.classifier_optimizer.zero_grad()
            #     self.yp = yp = self.classifier_model(self.cnn_model(x))
            #     self.loss = loss = self.classifier_loss_fn(yp, y)
            #     loss.backward()
            #     self.cnn_optimizer.step()
            #     self.classifier_optimizer.step()
            #     self.run_hooks('classifier_iter_end')
            # self.run_hooks('classifier_epoch_end')

            #
            # train the detective:
            #
            #     · freeze the classifier and CNN
            #     · loss = detective_loss
            #

            self.run_hooks('detective_epoch_begin')
            for i_iter, batch in enumerate(self.train_loader):
                if test_run and i_iter >= 3:
                    break
                self.run_hooks('detective_iter_begin')
                _, self.x, _, self.z = _, x, _, z = self.process_batch_fn(batch, self.device)
                self.cnn_optimizer.zero_grad()
                self.detective_optimizer.zero_grad()
                self.zp = zp = self.detective_model(self.cnn_model(x))
                self.loss = loss = self.detective_loss_fn(zp, z)
                loss.backward()
                self.detective_optimizer.step()
                self.run_hooks('detective_iter_end')
            self.run_hooks('detective_epoch_end')

            #
            # validate both:
            #
            #     · freeze everything
            #     · run both on the validation set
            #
            self.run_hooks('valid_epoch_begin')

            self.cnn_model.eval()
            self.classifier_model.eval()
            self.detective_model.eval()

            with torch.no_grad():
                for i_iter, batch in enumerate(self.valid_loader):
                    if test_run and i_iter >= 3:
                        break
                    self.run_hooks('valid_iter_begin')
                    _, self.x, self.y, self.z = _, x, y, z = self.process_batch_fn(batch, self.device)
                    cnn_out = self.cnn_model(x)
                    self.yp = yp = self.classifier_model(cnn_out)
                    self.zp = zp = self.detective_model(cnn_out)
                    self.run_hooks('valid_iter_end')

            self.run_hooks('valid_epoch_end')

            self.run_hooks('epoch_end')

            if test_run and self.i_epoch >= 3:
                break

            self.i_epoch += 1

        self.run_hooks('end')

        if test_run:
            warn('WARNING: test run finished')

    ### def find_lr(
    ###     self,
    ###     low=1e-10,
    ###     high=1e2,
    ###     n_iters=200,
    ###     n_reps=1,
    ###     save_csv_as='tmp/find_lr.csv',
    ###     save_plot_as='tmp/find_lr.pdf',
    ###     divergence_threshold=5,
    ###     smooth_factor=0.05,
    ###     fig_size=(20, 10),
    ### ):

    ###     def repeated(train_loader):
    ###         while True:
    ###             for batch in train_loader:
    ###                 yield batch

    ###     for param_group in self.classifier_optimizer.param_groups:
    ###         param_group['old_lr'] = param_group['lr']

    ###     history = {
    ###         'lr': [],
    ###         'running_loss': [],
    ###         'loss': [],
    ###     }

    ###     best_loss = np.inf
    ###     repeated_train_loader = repeated(self.train_loader)
    ###     self.classifier_model.to(device=self.device)
    ###     self.classifier_model.train()
    ###     running_loss = 0
    ###     residue_factor = 1
    ###     residue_next_term = 1
    ###     for i_iter in range(n_iters):
    ###         # exponential growth of lr
    ###         lr = low * (high / low)**(i_iter / (n_iters - 1))
    ###         for param_group in self.classifier_optimizer.param_groups:
    ###             param_group['lr'] = lr

    ###         x, y = self.process_batch_fn(next(repeated_train_loader), self.device)

    ###         self.classifier_optimizer.zero_grad()
    ###         yp = self.classifier_model(x)
    ###         # DEBUG: random guessing
    ###         # yp = (-1.1 * torch.ones((8, 1), dtype=torch.float)).to(self.device)
    ###         loss = self.classifier_loss_fn(yp, y)
    ###         loss.backward()
    ###         self.classifier_optimizer.step()

    ###         alpha = 1 / residue_factor
    ###         debug(f"{1 - alpha} - {alpha}")
    ###         running_loss = (1 - alpha) * running_loss + alpha * loss.item()
    ###         residue_next_term *= (1 - smooth_factor)
    ###         residue_factor += residue_next_term

    ###         history["lr"].append(lr)
    ###         history["loss"].append(loss.item())
    ###         history["running_loss"].append(running_loss)
    ###         info(' | '.join([
    ###             f"iter {i_iter}/{n_iters}",
    ###             f"running_loss {running_loss:.5f}",
    ###             f"lr {lr:.5f}",
    ###         ]))

    ###         # TODO: DEBUG
    ###         # print(torch.cat([yp, y], 1))
    ###         # TODO: DEBUG END

    ###         if running_loss > best_loss * divergence_threshold:
    ###             info(
    ###                 f"The running loss {running_loss} is more than {divergence_threshold}x "
    ###                 f"of the best loss {best_loss}. Stop early."
    ###             )
    ###             break

    ###         if running_loss < best_loss:
    ###             best_loss = running_loss

    ###     for param_group in self.classifier_optimizer.param_groups:
    ###         param_group['lr'] = param_group['old_lr']

    ###     if save_csv_as:
    ###         pd.DataFrame.from_records(history).to_csv(save_csv_as)
    ###         debug(f"Saved CSV as {save_csv_as}")

    ###     plt.figure(figsize=fig_size)
    ###     plt.plot(history["lr"], history["running_loss"])
    ###     plt.xscale("log")
    ###     plt.xlabel("Learning rate")
    ###     plt.ylabel("Running Loss")
    ###     if save_plot_as:
    ###         plt.savefig(save_plot_as)
    ###         debug(f"Saved plot as {save_plot_as}")
