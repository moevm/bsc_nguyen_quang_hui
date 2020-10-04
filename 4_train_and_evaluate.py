import torch
import matplotlib.pyplot as plt
from datetime import datetime
from torch.utils.data import Dataset, DataLoader
import json

# train the network with the prepared data
# save the best model for later use


use_cuda = torch.cuda.is_available()
print(use_cuda)
device = torch.device("cuda" if use_cuda else "cpu")

def f1_loss(y_true, y_pred, is_training=False) -> torch.Tensor:
    '''Calculate F1 score. Can work with gpu tensors
    
    The original implmentation is written by Michal Haltuf on Kaggle.
    
    Returns
    -------
    torch.Tensor
        `ndim` == 1. 0 <= val <= 1
    
    Reference
    ---------
    - https://www.kaggle.com/rejpalcz/best-loss-function-for-f1-score-metric
    - https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html#sklearn.metrics.f1_score
    - https://discuss.pytorch.org/t/calculating-precision-recall-and-f1-score-in-case-of-multi-label-classification/28265/6
    
    '''
    assert y_true.ndim == 1
    assert y_pred.ndim == 1 or y_pred.ndim == 2
    
    if y_pred.ndim == 2:
        y_pred = y_pred.argmax(dim=1)
        
    
    tp = (y_true * y_pred).sum()#.to(torch.float32)
    tn = ((1 - y_true) * (1 - y_pred)).sum()#.to(torch.cuda.float32)
    fp = ((1 - y_true) * y_pred).sum()#.to(torch.cuda.float32)
    fn = (y_true * (1 - y_pred)).sum()#.to(torch.cuda.float32)

    # print("TP  TN  FP  FN")
    # print(str(tp.item())+"  "+str(tn.item())+"  "+str(fp.item())+"  "+str(fn.item()))

    epsilon = 1e-7
    
    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)
    
    f1 = 2* (precision*recall) / (precision + recall + epsilon)
    f1.requires_grad = is_training
    return precision, recall, f1

class CustomDataset(torch.utils.data.Dataset):
    'Characterizes a dataset for PyTorch'

    def __init__(self, data):
        'Initialization'
        self.data = data
        self.inp = []
        self.out = []
        for entry in data:
            self.inp.append(torch.tensor(entry['inp'], device=device).float())
            self.out.append(torch.tensor(entry['out'], device=device).float())
            entry['inp']=None
            entry['out']=None
        self.inp = torch.stack(self.inp) #.to(device)
        self.out = torch.stack(self.out) #.to(device)

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.data)

    def __getitem__(self, index):
        'Generates one sample of data'
        return self.inp[index], self.out[index]

def evaluate(validating_dataloader):
    validating_iter = iter(validating_dataloader)
    val_losses = []
    predictions = []
    true_values = []
    with torch.no_grad():
        for t in range(140):
            # extract a batch from dataset
            x, y = next(validating_iter)
            if len(x)!=N:
                validating_iter = iter(validating_dataloader)
                x, y = next(validating_iter)
            # x = torch.stack(x, dim=-1).float().to(device)
            # y = torch.stack(y, dim=-1).float().to(device)
            model.eval()
            # Forward pass: compute predicted y by passing x to the model.
            y_pred = model(x)
            y_bin = (y_pred > 0.5).float()
            predictions.append(y_bin.squeeze())
            true_values.append(y.squeeze())
            # Compute and print loss.
            # loss = loss_fn(y_pred, y)
            # print("--------------")
            # print(y_pred.squeeze())
            # print(y_bin.squeeze())
            # print(y.squeeze())
            # print((y_bin == y).sum().item())
            # print(loss)
            # val_losses.append(loss.item())
    return val_losses, predictions, true_values


with open('training-cl-25f-tfidf.txt') as json_file:
    dataset1 = json.load(json_file)
    for d in dataset1:
        d['inp'] = d['inp'][:3]+d['inp'][6:]
    
# with open('training-cl2.txt') as json_file:
#     dataset1 = dataset1 + json.load(json_file)
# with open('training-cl3.txt') as json_file:
#     dataset1 = dataset1 +json.load(json_file)
# with open('training-cl4.txt') as json_file:
#     dataset1 = dataset1 +json.load(json_file)
with open('test-cl-25f-tfidf.txt') as json_file:
    dataset2 = json.load(json_file)
    for d in dataset2:
        d['inp'] = d['inp'][:3]+d['inp'][6:]
# with open('test-cl2.txt') as json_file:
#     dataset2 = dataset2+json.load(json_file)
# with open('test-cl3.txt') as json_file:
#     dataset2 = dataset2+json.load(json_file)
# with open('test-cl4.txt') as json_file:
#     dataset2 = dataset2+json.load(json_file)

# preparation of training set and test set
# with data_loader
training_set = CustomDataset(dataset1)
validating_set = CustomDataset(dataset2)

# model creation
D_in = 29  # kw_length, kw_count, avg_cosine, max_cosine
# H1 = 150
# H2 = 250
# H3 = 150
H=450
D_out = 1
N = 1024

training_dataloader = DataLoader(training_set, batch_size=N,
                                 shuffle=True, num_workers=0)

validating_dataloader = DataLoader(validating_set, batch_size=N,
                                   shuffle=True, num_workers=0)

model = torch.nn.Sequential(
    torch.nn.Linear(D_in, H),
    torch.nn.LeakyReLU(),
    torch.nn.Dropout(0.2),
    torch.nn.Linear(H, H),
    torch.nn.LeakyReLU(),
    torch.nn.Dropout(0.2),
    torch.nn.Linear(H, H),
    torch.nn.LeakyReLU(),
    torch.nn.Dropout(0.2),
    # torch.nn.Linear(H, H),
    # torch.nn.LeakyReLU(),
    # torch.nn.Dropout(0.2),
    # torch.nn.Linear(H, H),
    # torch.nn.LeakyReLU(),
    # torch.nn.Dropout(0.2),
    torch.nn.Linear(H, D_out),
    torch.nn.Sigmoid(),
)

model.to(device)

def init_weights(m):
    if type(m) == torch.nn.Linear:
        torch.nn.init.normal_(m.weight, 0.0, 0.2)

# model.apply(init_weights)


loss_fn = torch.nn.BCELoss().cuda()
# loss_fn = torch.nn.MSELoss(reduction='sum')

learning_rate = 1e-3
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5000, gamma=0.5)

training_iter = iter(training_dataloader)

# evaluation before training
val_losses, predictions, true_values = evaluate(validating_dataloader)

predictions = torch.cat(predictions)
true_values = torch.cat(true_values)
total = predictions.size(0)
correct = (predictions == true_values).sum().item()
precision, recall, f1 = f1_loss(true_values,predictions)
print("Precision:  "+str(precision.item()))
print("Recall:     "+str(recall.item()))
print("F1:         "+str(f1.item()))
print("Accuracy:   "+str(float(correct)/float(total)))

val_losses = []
total_loss = 0
max_f1 = 0
# train
for t in range(50000):

    # Create random Tensors to hold inputs and outputs
    # TODO: extract a batch from dataset
    x, y = next(training_iter)
    if len(x)!=N:
        training_iter = iter(training_dataloader)
        x, y = next(training_iter)
    # x = torch.stack(x, dim=-1).to(device).float()
    # y = torch.stack(y, dim=-1).to(device).float()

    model.train()
    # Forward pass: compute predicted y by passing x to the model.
    y_pred = model(x)
    # y_pred[y_pred!=y_pred] = 0 
    # print(str(y_pred.tolist()))

    # Compute and print loss.
    loss = loss_fn(y_pred, y)
    # total_loss +=loss.item()
    if t % 500 == 499:
        # val_losses.append(total_loss/100)
        # total_loss=0
        print(t, loss.item(),optimizer.param_groups[0]['lr'])

        # evaluation
        _ , predictions, true_values = evaluate(validating_dataloader)

        predictions = torch.cat(predictions)
        true_values = torch.cat(true_values)
        total = predictions.size(0)
        correct = (predictions == true_values).sum().item()
        precision, recall, f1 = f1_loss(true_values,predictions)
        print("--------------")
        print("Precision:  "+str(precision.item()))
        print("Recall:     "+str(recall.item()))
        print("F1:         "+str(f1.item()))
        print("Accuracy  : "+str(float(correct)/float(total)))
        if(f1.item()>max_f1 and t>10000):
            max_f1 = f1.item()
            print("--------------SAVED")
            # for param in model.parameters():
            #     print(param.data)
            # torch.save(model, "./latest.model")
            torch.save(model.state_dict(),'./state_dict.pt')


    # Before the backward pass, use the optimizer object to zero all of the
    # gradients for the variables it will update (which are the learnable
    # weights of the model). This is because by default, gradients are
    # accumulated in buffers( i.e, not overwritten) whenever .backward()
    # is called. Checkout docs of torch.autograd.backward for more details.
    optimizer.zero_grad()

    # Backward pass: compute gradient of the loss with respect to model
    # parameters
    loss.backward()

    # Calling the step function on an Optimizer makes an update to its
    # parameters
    optimizer.step()
    scheduler.step()

# plt.plot(val_losses)

# plt.show()

