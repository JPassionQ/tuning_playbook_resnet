import matplotlib.pyplot as plt

def plot_loss_curve(epochs, train_losses, val_losses, save_path):
    plt.figure()
    plt.plot(epochs, train_losses, label='Train Loss')
    plt.plot(epochs, val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Train/Val Loss Curve')
    plt.savefig(save_path)
    plt.close()
