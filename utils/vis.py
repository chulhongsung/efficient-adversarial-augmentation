import matplotlib.pyplot as plt
import torch
from sklearn.metrics import precision_score, recall_score, roc_auc_score, roc_curve, auc, confusion_matrix, precision_recall_curve, PrecisionRecallDisplay, average_precision_score

def plot_recall_precision(model, data_loader, device):

    model.eval()
    y_true = []
    y_scores = []

    for _, batch in enumerate(data_loader):
        batch = batch.to(device)

        with torch.no_grad():
            pred = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)

        y_true.append(batch.y.view(pred.shape))
        y_scores.append(pred)

    y_true = torch.cat(y_true, dim = 0).cpu().numpy()
    y_scores = torch.cat(y_scores, dim = 0).cpu().numpy()

    precision, recall, threshold = precision_recall_curve(y_true.ravel(), y_scores.ravel())
    average_precision = average_precision_score(y_true.ravel(), y_scores.ravel())

    _, ax = plt.subplots(figsize=(7, 8))

    display = PrecisionRecallDisplay(
        recall=recall,
        precision=precision,
        average_precision=average_precision
    )

    display.plot(ax=ax)

    return precision, recall, threshold

def plot_recall_precision2(model, data_loader, device):

    model.eval()
    y_true = []
    y_scores = []

    for _, batch in enumerate(data_loader):
        batch_x, batch_y = batch

        with torch.no_grad():
            pred = model(batch_x)

        y_true.append(batch_y.view(pred.shape))
        y_scores.append(pred)

    y_true = torch.cat(y_true, dim = 0).cpu().numpy()
    y_scores = torch.cat(y_scores, dim = 0).cpu().numpy()

    precision, recall, threshold = precision_recall_curve(y_true.ravel(), y_scores.ravel())
    average_precision = average_precision_score(y_true.ravel(), y_scores.ravel())

    _, ax = plt.subplots(figsize=(7, 8))

    display = PrecisionRecallDisplay(
        recall=recall,
        precision=precision,
        average_precision=average_precision
    )

    display.plot(ax=ax)

    return precision, recall, threshold