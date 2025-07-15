import matplotlib.pyplot as plt
import numpy as np

def smooth_curve(values, window_size=5):
    """简单滑动平均平滑曲线，避免两端异常"""
    if window_size < 2:
        return np.array(values)
    values = np.array(values)
    kernel = np.ones(window_size) / window_size
    # 只对中间部分做平滑
    smooth = np.convolve(values, kernel, mode='valid')
    # 两端补原始值
    pad = (window_size - 1) // 2
    left = values[:pad]
    right = values[-pad:] if pad > 0 else np.array([])
    return np.concatenate([left, smooth, right])

def ema_smooth_curve(values, alpha=0.8):
    """
    使用指数移动平均（EMA）平滑曲线
    :param values: 输入序列
    :param alpha: 平滑系数，0<alpha<1，越大越平滑
    :return: 平滑后的序列
    """
    values = np.array(values)
    if len(values) == 0:
        return values
    smoothed = np.zeros_like(values, dtype=float)
    smoothed[0] = values[0]
    for i in range(1, len(values)):
        smoothed[i] = alpha * smoothed[i-1] + (1 - alpha) * values[i]
    return smoothed

def plot_loss_curve(total_steps, train_losses, val_losses, save_path):
    plt.figure()
    # 原始曲线
    xmin = 0
    xmax = total_steps
    train_steps = np.linspace(xmin, xmax, len(train_losses))
    val_steps = np.linspace(xmin, xmax, len(val_losses))
    plt.plot(train_steps, train_losses, label='Train Loss (raw)', alpha=0.5, linewidth=1)
    plt.plot(val_steps, val_losses, label='Val Loss (raw)', alpha=0.5, linewidth=1)
    # 平滑曲线
    train_losses_smooth = ema_smooth_curve(train_losses)
    val_losses_smooth = ema_smooth_curve(val_losses)
    plt.plot(train_steps, train_losses_smooth, label='Train Loss (smoothed)', linewidth=1.8)
    plt.plot(val_steps, val_losses_smooth, label='Val Loss (smoothed)', linewidth=1.8)
    
    plt.xlim(xmin, xmax)
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Train/Val Loss Curve')
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()

def plot_loss_compare_curve(total_steps, losses, save_path, loss_type):
    plt.figure()
    # 原始曲线
    xmin = 0
    xmax = total_steps
    true_loss_steps = 0
    for _, v in losses.items():
        true_loss_steps = len(v)
        break
    train_steps = np.linspace(xmin, xmax, true_loss_steps)
    for type_, loss in losses.items():
        losses_smooth = ema_smooth_curve(losses[type_])
        plt.plot(train_steps, losses_smooth, label=f"{type_}", linewidth=1.2)
    plt.xlim(xmin, xmax)
    plt.xlabel('Step')
    plt.ylabel(f'{loss_type} Loss')
    plt.legend()
    plt.title(f'{loss_type} Loss Curve')
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()

def plot_acc_compare_curve(total_steps, valid_accs, save_path):
    plt.figure()
    # 原始曲线
    xmin = 0
    xmax = total_steps
    true_acc_steps = 0
    for _, v in valid_accs.items():
        true_acc_steps = len(v)
        break
    train_steps = np.linspace(xmin, xmax, true_acc_steps)
    for type_, acc in valid_accs.items():
        acc_smooth = ema_smooth_curve(acc)
        plt.plot(train_steps, acc_smooth, label=f"{type_}", linewidth=1.2)
    plt.xlim(xmin, xmax)
    plt.xlabel('Step')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.title('Validation Accuracy Curve')
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()

def plot_accuracy_curve(total_steps, val_accuracies, save_path):
    """
    绘制验证集准确率曲线
    :param epochs: epoch编号序列
    :param val_accuracies: 每个epoch的验证集准确率（百分比）
    :param save_path: 保存图片的路径
    """
    plt.figure()
    xmin = 0
    xmax = total_steps
    valid_steps = np.linspace(xmin, xmax, len(val_accuracies))
    # 原始曲线，减小线宽
    plt.plot(valid_steps, val_accuracies, label='Val Accuracy (raw)', alpha=0.5, linewidth=1)
    # 平滑曲线，减小线宽
    val_accuracies_smooth = ema_smooth_curve(val_accuracies)
    plt.plot(valid_steps, val_accuracies_smooth, label='Val Accuracy (smoothed)', linewidth=1.8)
    plt.xlabel('Steps')
    plt.ylabel('Accuracy (%)')
    plt.title('Validation Accuracy Curve')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()

def plot_hyperparam_curve(hyperparam, acc, save_path, x_name):
    """
    绘制基本超参数轴图
    :param hyperparam: 超参数取值序列
    :param acc: 对应的准确率序列
    """

    hyperparam = np.array(hyperparam)
    acc = np.array(acc)

    # infeasible trials: acc == 10%
    infeasible_mask = acc == 10
    # feasible trials: acc != 10%
    feasible_mask = ~infeasible_mask

    # Top 2 trials among feasible
    feasible_acc = acc[feasible_mask]
    feasible_hyper = hyperparam[feasible_mask]
    if len(feasible_acc) >= 2:
        top2_idx = np.argsort(feasible_acc)[-2:]
    else:
        top2_idx = np.argsort(feasible_acc)

    # Masks for top2 and other feasible
    top2_mask = np.zeros_like(feasible_acc, dtype=bool)
    top2_mask[top2_idx] = True
    other_feasible_mask = ~top2_mask

    plt.figure()

    # Plot infeasible
    plt.scatter(hyperparam[infeasible_mask], acc[infeasible_mask], 
                marker='x', color='red', label='infeasible trials')

    # Plot other feasible
    plt.scatter(feasible_hyper[other_feasible_mask], feasible_acc[other_feasible_mask], 
                marker='o', color='blue', label='feasible trials')

    # Plot top2
    plt.scatter(feasible_hyper[top2_mask], feasible_acc[top2_mask], 
                marker='*', color='green', s=150, label='Top 2 trials')

    plt.xlabel(x_name)
    plt.ylabel('acc (%)')
    plt.title(f'{x_name} vs. acc')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()

def plot_hyperparam_2d_curve(lr_list, momentum_list, acc, save_path):
    """
    绘制准确率随着两个超参数（学习率和动量因子）变化的三维散点图
    :param lr_list: 学习率序列
    :param momentum_list: 动量因子序列
    :param acc: 对应的准确率序列
    :param save_path: 保存图片的路径
    """
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    lr_arr = np.array(lr_list)
    momentum_arr = np.array(momentum_list)
    acc_arr = np.array(acc)

    infeasible_mask = acc_arr == 10
    feasible_mask = ~infeasible_mask

    feasible_acc = acc_arr[feasible_mask]
    feasible_lr = lr_arr[feasible_mask]
    feasible_momentum = momentum_arr[feasible_mask]
    if len(feasible_acc) >= 2:
        top2_idx = np.argsort(feasible_acc)[-2:]
    else:
        top2_idx = np.argsort(feasible_acc)

    top2_mask = np.zeros_like(feasible_acc, dtype=bool)
    top2_mask[top2_idx] = True
    other_feasible_mask = ~top2_mask

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot infeasible
    ax.scatter(lr_arr[infeasible_mask], momentum_arr[infeasible_mask], acc_arr[infeasible_mask],
               marker='x', color='red', label='infeasible trials')

    # Plot other feasible
    ax.scatter(feasible_lr[other_feasible_mask], feasible_momentum[other_feasible_mask], feasible_acc[other_feasible_mask],
               marker='o', color='blue', label='feasible trials')

    # Plot top2
    ax.scatter(feasible_lr[top2_mask], feasible_momentum[top2_mask], feasible_acc[top2_mask],
               marker='*', color='green', s=150, label='Top 2 trials')

    ax.set_xlabel('lr')
    ax.set_ylabel('momentum')
    ax.set_zlabel('acc (%)')
    ax.set_title('lr & momentum vs. acc (3D)')

    # Set axis limits according to actual data range
    ax.set_xlim(np.min(lr_arr), np.max(lr_arr))
    ax.set_ylim(np.min(momentum_arr), np.max(momentum_arr))

    ax.legend()
    plt.savefig(save_path)
    plt.close()

