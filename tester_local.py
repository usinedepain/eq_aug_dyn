import torch
from tqdm import tqdm

def test(model, loader, criterion, device):

    running_loss = 0.0
    running_acc = 0.0
    
    model.eval()
    with torch.no_grad():

        for data in tqdm(loader, desc='Testing progress', leave=False):
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs, _ = model(inputs)
            loss = criterion(outputs,labels)
            acc = outputs.argmax(1,keepdim=True).eq(labels.view_as(outputs.argmax(1,keepdim=True))).sum().float() / labels.shape[0]

            running_loss += loss.item()
            running_acc += acc.item()
    return running_loss/len(loader), running_acc/len(loader)