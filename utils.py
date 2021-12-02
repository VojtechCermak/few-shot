from sklearn.neighbors import KNeighborsClassifier
import torch

def get_embeddings(model, dl_test, dl_train):
    '''
    Calculate embeddings on train and valid datasets.
    '''
    model = model.eval()
    model = model.cuda()

    x_train = []
    y_train = []
    with torch.no_grad():
        for x, y in dl_train:
            x = x.cuda()
            x_train.append(model.embedding(x).detach().cpu())
            y_train.append(y)
    x_train = torch.cat(x_train)
    y_train = torch.cat(y_train)
    
    x_test = []
    y_test = []
    with torch.no_grad():
        for x, y in dl_test:
            x = x.cuda()
            x_test.append(model.embedding(x).detach().cpu())
            y_test.append(y)
    x_test = torch.cat(x_test)
    y_test = torch.cat(y_test)
    
    return x_train, y_train, x_test, y_test


def evaluate(x_train, y_train, x_test, y_test, k=10):
    '''
    Evaluate embeddings using k-NN classifier.
    '''
    # Train k-nn classifier
    classifier = KNeighborsClassifier(n_neighbors=k)
    classifier.fit(x_train.cpu().numpy(), y_train.cpu().numpy())

    # Accuracy on all test samples
    y_pred = classifier.predict(x_test.cpu().numpy())
    acc = sum(y_pred == y_test.cpu().numpy()) / len(y_pred)
    return acc

