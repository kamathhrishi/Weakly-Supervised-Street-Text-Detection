import torch


def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model = checkpoint["model"]
    model.load_state_dict(checkpoint["state_dict"])
    for parameter in model.parameters():
        parameter.requires_grad = True

    model.eval()
    return model


def save_model(model, optimizer, name="local_model.pth"):

    checkpoint = {
        "model": model,
        "state_dict": model.state_dict(),
    }

    torch.save(checkpoint, name)


def evaluate(model, loader, label):

    model.eval()

    correct = 0.0
    total = 0.0

    for data, target in loader:

        outputs = model(data.double())
        correct += (torch.argmax(outputs, axis=1) == target).sum()
        total += data.shape[0]
        # print(float(correct/total)*100)

    print(label, float(correct / total) * 100)
    return float(correct / total) * 100
