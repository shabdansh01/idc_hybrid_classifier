def accuracy(preds, labels):
    return ((preds > 0.5) == labels).float().mean()
