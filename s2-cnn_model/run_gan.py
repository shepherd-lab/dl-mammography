def gan(do_test_run=False, **kwargs):
    max_epochs = config['max_epochs']
    device = config['device']
    pwd = config['pwd']

    model = get_model(config)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    loss_fn = nn.BCEWithLogitsLoss()
    loader_pair = get_loader_pair(config)

    plugins = [
        Timestamp(),
        TrainingMetrics({
            'loss': loss,
            'acc': acc,
        }, residual_factor=max(1 - config['train_batch_size'] / 10000, 0)),
        ValidationMetrics({
            'loss': loss,
            'acc': acc,
            'std': std,
            'auc': auc,
        }),
        ReduceLROnPlateau(),
        Checkpoint(),
        Messages(),
    ]

    print(f"config =\n{config}")

    config.update(kwargs)

    if os.path.exists(config['pwd']):
        if input(f"{config['pwd']} already exists, are you sure? [y/N]: ") != 'y':
            return

    if test_run:
        warn('WARNING: this is a test run')

    self.model.to(device=self.device)
    while self.i_epoch < self.max_epochs:
        self.run_hooks('epoch_begin')

        self.model.train()
        for i_iter, batch in enumerate(self.train_loader):
            if test_run and i_iter >= 3:
                break
            self.run_hooks('train_iter_begin')
            self.x, self.y = x, y = self.process_batch_fn(batch, self.device)
            self.optimizer.zero_grad()
            self.yp = yp = self.model(x)
            self.loss = loss = self.loss_fn(yp, y)
            loss.backward()
            self.optimizer.step()
            self.run_hooks('train_iter_end')
        self.run_hooks('train_epoch_end')

        self.run_hooks('valid_epoch_begin')
        self.model.eval()
        with torch.no_grad():
            for i_iter, batch in enumerate(self.valid_loader):
                if test_run and i_iter >= 3:
                    break
                self.run_hooks('valid_iter_begin')
                self.x, self.y = x, y = self.process_batch_fn(batch, self.device)
                self.yp = yp = self.model(x)
                self.run_hooks('valid_iter_end')

        self.run_hooks('epoch_end')

        if test_run and self.i_epoch >= 10:
            break

        self.i_epoch += 1

    self.run_hooks('end')

    if test_run:
        warn('WARNING: test run finished')

    pass
