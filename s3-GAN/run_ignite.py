from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator

create_supervised_trainer(
    model,
    optimizer,
    loss_fn,
    device=None,
    non_blocking=False,
    prepare_batch=_prepare_batch,
    output_transform=lambda x, y, y_pred, loss: loss.item(),
)

create_supervised_evaluator(
    model,
    metrics={},
    device=None,
    non_blocking=False,
    prepare_batch=_prepare_batch,
    output_transform=lambda x, y, y_pred: (y_pred, y),
)


def step(engine, batch):
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
    self.run_hooks('train_epoch_begin')
    for i_iter, batch in enumerate(self.train_loader):
        if test_run and i_iter >= 3:
            break
        self.run_hooks('train_iter_begin')
        self.x, self.y, _ = x, y, _ = self.process_batch_fn(batch, self.device)
        self.cnn_optimizer.zero_grad()
        self.classifier_optimizer.zero_grad()
        self.yp = yp = self.classifier_model(self.cnn_model(x))
        self.loss = loss = self.classifier_loss_fn(yp, y)
        loss.backward()
        self.cnn_optimizer.step()
        self.classifier_optimizer.step()
        self.run_hooks('train_iter_end')
    self.run_hooks('train_epoch_end')

    #
    # train the detective:
    #
    #     · freeze the classifier and CNN
    #     · loss = detective_loss
    #
    self.run_hooks('detective_train_epoch_begin')
    for i_iter, batch in enumerate(self.train_loader):
        if test_run and i_iter >= 3:
            break
        self.run_hooks('detective_train_iter_begin')
        self.x, _, self.z = x, _, z = self.process_batch_fn(batch, self.device)
        self.cnn_optimizer.zero_grad()
        self.detective_optimizer.zero_grad()
        self.zp = zp = self.detective_model(self.cnn_model(x))
        self.loss = loss = self.detective_loss_fn(zp, z)
        loss.backward()
        self.detective_optimizer.step()
        self.run_hooks('detective_train_iter_end')
    self.run_hooks('detective_train_epoch_end')

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
            self.x, self.y, self.z = x, y, z = self.process_batch_fn(batch, self.device)
            cnn_out = self.cnn_model(x)
            self.yp = yp = self.classifier_model(cnn_out)
            self.zp = zp = self.detective_model(cnn_out)
            self.run_hooks('valid_iter_end')

    self.run_hooks('epoch_end')

    if test_run and self.i_epoch >= 10:
        break

    self.i_epoch += 1


if __name__ == '__main__':
    fire.Fire()
