import matplotlib.pyplot as plt

def plot_loss_curve(epochs, train_losses, val_losses, save_path):
    plt.figure()
    plt.plot(epochs, train_losses, label='Train Loss')
    plt.plot(epochs, val_losses, label='Val Loss')
    plt.xlabel('steps')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Train/Val Loss Curve')
    plt.savefig(save_path)
    plt.close()

def plot_accuracy_curve(epochs, val_accuracies, save_path):
    """
    绘制验证集准确率曲线
    :param epochs: epoch编号序列
    :param val_accuracies: 每个epoch的验证集准确率（百分比）
    :param save_path: 保存图片的路径
    """
    plt.figure()
    plt.plot(epochs, val_accuracies, label='Val Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Validation Accuracy Curve')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()
