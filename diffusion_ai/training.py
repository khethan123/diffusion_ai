import torch


def accuracy(out, yb):
    """
    Computes the accuracy of the model's output.

    Params:
        out (torch.Tensor): The model's output.
        yb (torch.Tensor): The target labels.

    Returns:
        float: The accuracy of the model's output.
    """
    return (out.argmax(dim=1) == yb).float().mean()


def report(loss, preds, yb):
    """
    Prints the loss and accuracy.

    Params:
        loss (float): The loss value.
        preds (torch.Tensor): The model's predictions.
        yb (torch.Tensor): The target labels.
    """
    print(f"{loss:.2f}, {accuracy(preds, yb):.2f}")


class Dataset:
    """
    A simple dataset class for PyTorch.

    Params:
        x (torch.Tensor): The input data.
        y (torch.Tensor): The target labels.
    """

    def __init__(self, x, y):
        self.x, self.y = x, y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        return self.x[i], self.y[i]


def fit(epochs, model, loss_func, opt, train_dl, valid_dl):
    """
    Trains the model for a specified number of epochs.

    Params:
        epochs (int): The number of epochs to train for.
        model (torch.nn.Module): The model to train.
        loss_func (function): The loss function.
        opt (torch.optim.Optimizer): The optimizer.
        train_dl (torch.utils.data.DataLoader): The training data loader.
        valid_dl (torch.utils.data.DataLoader): The validation data loader.

    Returns:
        tuple: The final loss and accuracy.
    """
    for epoch in range(epochs):
        model.train()
        for xb, yb in train_dl:
            loss = loss_func(model(xb), yb)
            loss.backward()
            opt.step()
            opt.zero_grad()

        model.eval()
        with torch.no_grad():
            tot_loss, tot_acc, count = 0.0, 0.0, 0
            for xb, yb in valid_dl:
                pred = model(xb)
                n = len(xb)
                count += n
                tot_loss += loss_func(pred, yb).item() * n
                tot_acc += accuracy(pred, yb).item() * n
        print(epoch, tot_loss / count, tot_acc / count)
    return tot_loss / count, tot_acc / count
