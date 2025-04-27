import numpy as np
import os
import struct
import urllib.request
import gzip
import time
import matplotlib.pyplot as plt 

# 定义改进版AlexNet类
class AlexNet:
    def __init__(self, input_shape=(1, 28, 28), num_classes=10):
        self.input_shape = input_shape
        self.num_classes = num_classes
        
        # 初始化存储容器
        self.params = {}
        self.activations = {}
        self.bn_params = {}
        self.bn_cache = {}
        
        # 初始化权重
        self._initialize_weights()
        
    def _initialize_weights(self):
        # 使用更好的权重初始化 - He初始化
        # 卷积层1: (32, 1, 5, 5) - 使用更大的卷积核
        n_in = self.input_shape[0] * 5 * 5
        self.params['W1'] = np.random.randn(32, self.input_shape[0], 5, 5) * np.sqrt(2.0 / n_in)
        self.params['b1'] = np.zeros(32)
        
        # 批归一化层1参数
        self.params['gamma1'] = np.ones(32)
        self.params['beta1'] = np.zeros(32)
        self.bn_params['mean1'] = np.zeros(32)
        self.bn_params['var1'] = np.ones(32)
        
        # 卷积层2: (64, 32, 5, 5)
        n_in = 32 * 5 * 5
        self.params['W2'] = np.random.randn(64, 32, 5, 5) * np.sqrt(2.0 / n_in)
        self.params['b2'] = np.zeros(64)
        
        # 批归一化层2参数
        self.params['gamma2'] = np.ones(64)
        self.params['beta2'] = np.zeros(64)
        self.bn_params['mean2'] = np.zeros(64)
        self.bn_params['var2'] = np.ones(64)
        
        # 卷积层3: (128, 64, 3, 3)
        n_in = 64 * 3 * 3
        self.params['W3'] = np.random.randn(128, 64, 3, 3) * np.sqrt(2.0 / n_in)
        self.params['b3'] = np.zeros(128)
        
        # 批归一化层3参数
        self.params['gamma3'] = np.ones(128)
        self.params['beta3'] = np.zeros(128)
        self.bn_params['mean3'] = np.zeros(128)
        self.bn_params['var3'] = np.ones(128)
        
        # 全连接层1 (特征图到隐藏层)
        # 我们将在前向传播时动态确定正确的特征尺寸
        # 暂时使用一个默认值
        final_conv_size = 2048
        
        # 全连接层1
        n_in = final_conv_size
        self.params['W4'] = np.random.randn(final_conv_size, 1024) * np.sqrt(2.0 / n_in)
        self.params['b4'] = np.zeros(1024)
        
        # 批归一化层4参数
        self.params['gamma4'] = np.ones(1024)
        self.params['beta4'] = np.zeros(1024)
        self.bn_params['mean4'] = np.zeros(1024)
        self.bn_params['var4'] = np.ones(1024)
        
        # 全连接层2 (输出层)
        n_in = 1024
        self.params['W5'] = np.random.randn(1024, self.num_classes) * np.sqrt(2.0 / n_in)
        self.params['b5'] = np.zeros(self.num_classes)
    
    def forward(self, X, training=True):
        """
        前向传播
        
        参数:
            X: 输入数据，形状 (N, C, H, W)
            training: 是否处于训练模式
            
        返回:
            输出概率，形状 (N, num_classes)
        """
        # 存储中间激活值
        self.activations = {}
        self.activations['X'] = X
        
        # 第一卷积层 + 批归一化 + ReLU + 池化
        Z1 = self._conv_forward(X, self.params['W1'], self.params['b1'], stride=1, pad=2)
        BN1 = self._batch_norm_forward(Z1, self.params['gamma1'], self.params['beta1'], 
                                      self.bn_params['mean1'], self.bn_params['var1'], 
                                      training=training, layer_idx=1)
        A1 = self._relu(BN1)
        P1 = self._max_pool_forward(A1, pool_size=2, stride=2)
        self.activations['Z1'], self.activations['BN1'], self.activations['A1'], self.activations['P1'] = Z1, BN1, A1, P1
        
        # 第二卷积层 + 批归一化 + ReLU + 池化
        Z2 = self._conv_forward(P1, self.params['W2'], self.params['b2'], stride=1, pad=2)
        BN2 = self._batch_norm_forward(Z2, self.params['gamma2'], self.params['beta2'], 
                                      self.bn_params['mean2'], self.bn_params['var2'], 
                                      training=training, layer_idx=2)
        A2 = self._relu(BN2)
        P2 = self._max_pool_forward(A2, pool_size=2, stride=2)
        self.activations['Z2'], self.activations['BN2'], self.activations['A2'], self.activations['P2'] = Z2, BN2, A2, P2
        
        # 第三卷积层 + 批归一化 + ReLU + 池化
        Z3 = self._conv_forward(P2, self.params['W3'], self.params['b3'], stride=1, pad=1)
        BN3 = self._batch_norm_forward(Z3, self.params['gamma3'], self.params['beta3'], 
                                      self.bn_params['mean3'], self.bn_params['var3'], 
                                      training=training, layer_idx=3)
        A3 = self._relu(BN3)
        P3 = self._max_pool_forward(A3, pool_size=2, stride=2)
        self.activations['Z3'], self.activations['BN3'], self.activations['A3'], self.activations['P3'] = Z3, BN3, A3, P3
        
        # 展平
        F = P3.reshape(P3.shape[0], -1)
        self.activations['F'] = F
        
        # 检查展平后的特征大小并更新W4维度（如果需要）
        if training and F.shape[1] != self.params['W4'].shape[0]:
            # 更新W4和b4形状以匹配特征
            n_in = F.shape[1]
            self.params['W4'] = np.random.randn(n_in, 1024) * np.sqrt(2.0 / n_in)
            self.params['b4'] = np.zeros(1024)
            
            # 更新BN参数
            self.params['gamma4'] = np.ones(1024)
            self.params['beta4'] = np.zeros(1024)
            
        
        # 全连接层1 + 批归一化 + ReLU + Dropout
        Z4 = F.dot(self.params['W4']) + self.params['b4']
        BN4 = self._batch_norm_forward_fc(Z4, self.params['gamma4'], self.params['beta4'], 
                                        self.bn_params['mean4'], self.bn_params['var4'], 
                                        training=training, layer_idx=4)
        A4 = self._relu(BN4)
        if training:
            # 训练时使用dropout
            A4 = self._dropout(A4, keep_prob=0.5)
        self.activations['Z4'], self.activations['BN4'], self.activations['A4'] = Z4, BN4, A4
        
        # 全连接层2 (输出层) + Softmax
        Z5 = A4.dot(self.params['W5']) + self.params['b5']
        A5 = self._softmax(Z5)
        self.activations['Z5'], self.activations['A5'] = Z5, A5
        
        return A5
    
    def _batch_norm_forward(self, X, gamma, beta, running_mean, running_var, training=True, layer_idx=0, eps=1e-5, momentum=0.9):
        """
        批归一化前向传播（卷积层）
        """
        N, C, H, W = X.shape
        
        if training:
            # 计算均值和方差 (对每个通道)
            mu = np.mean(X, axis=(0, 2, 3), keepdims=True)
            var = np.var(X, axis=(0, 2, 3), keepdims=True)
            
            # 更新运行时均值和方差
            self.bn_params[f'mean{layer_idx}'] = momentum * running_mean + (1 - momentum) * mu.reshape(-1)
            self.bn_params[f'var{layer_idx}'] = momentum * running_var + (1 - momentum) * var.reshape(-1)
            
            # 归一化
            X_norm = (X - mu) / np.sqrt(var + eps)
            
            # 缩放和平移
            out = gamma.reshape(1, C, 1, 1) * X_norm + beta.reshape(1, C, 1, 1)
            
            # 存储反向传播所需的中间值
            self.bn_cache[layer_idx] = (X, X_norm, mu, var, gamma, beta, eps)
        else:
            # 测试时使用运行时均值和方差
            X_norm = (X - running_mean.reshape(1, C, 1, 1)) / np.sqrt(running_var.reshape(1, C, 1, 1) + eps)
            out = gamma.reshape(1, C, 1, 1) * X_norm + beta.reshape(1, C, 1, 1)
        
        return out
    
    def _batch_norm_forward_fc(self, X, gamma, beta, running_mean, running_var, training=True, layer_idx=0, eps=1e-5, momentum=0.9):
        """
        批归一化前向传播（全连接层）
        """
        if training:
            # 计算均值和方差
            mu = np.mean(X, axis=0)
            var = np.var(X, axis=0)
            
            # 更新运行时均值和方差
            self.bn_params[f'mean{layer_idx}'] = momentum * running_mean + (1 - momentum) * mu
            self.bn_params[f'var{layer_idx}'] = momentum * running_var + (1 - momentum) * var
            
            # 归一化
            X_norm = (X - mu) / np.sqrt(var + eps)
            
            # 缩放和平移
            out = gamma * X_norm + beta
            
            # 存储反向传播所需的中间值
            self.bn_cache[layer_idx] = (X, X_norm, mu, var, gamma, beta, eps)
        else:
            # 测试时使用运行时均值和方差
            X_norm = (X - running_mean) / np.sqrt(running_var + eps)
            out = gamma * X_norm + beta
        
        return out
    
    def _conv_forward(self, X, W, b, stride=1, pad=0):
        """
        卷积前向传播
        """
        n_filters, d_filter, h_filter, w_filter = W.shape
        n_x, d_x, h_x, w_x = X.shape
        
        # 计算输出维度
        h_out = (h_x - h_filter + 2 * pad) // stride + 1
        w_out = (w_x - w_filter + 2 * pad) // stride + 1
        
        # 初始化输出
        out = np.zeros((n_x, n_filters, h_out, w_out))
        
        # 填充
        if pad > 0:
            X_padded = np.pad(X, ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode='constant')
        else:
            X_padded = X
        
        # 卷积操作
        for i in range(h_out):
            for j in range(w_out):
                h_start = i * stride
                w_start = j * stride
                X_slice = X_padded[:, :, h_start:h_start + h_filter, w_start:w_start + w_filter]
                
                # 转换维度以便进行批量计算
                X_slice_reshaped = X_slice.reshape(n_x, d_x * h_filter * w_filter)
                W_reshaped = W.reshape(n_filters, d_filter * h_filter * w_filter)
                
                # 计算输出
                out[:, :, i, j] = np.dot(X_slice_reshaped, W_reshaped.T) + b
        
        return out
    
    def _max_pool_forward(self, X, pool_size=2, stride=2):
        """
        最大池化前向传播
        """
        n, d, h, w = X.shape
        
        # 计算输出维度
        h_out = (h - pool_size) // stride + 1
        w_out = (w - pool_size) // stride + 1
        
        # 初始化输出
        out = np.zeros((n, d, h_out, w_out))
        
        # 最大池化操作
        for i in range(h_out):
            for j in range(w_out):
                h_start = i * stride
                w_start = j * stride
                X_slice = X[:, :, h_start:h_start + pool_size, w_start:w_start + pool_size]
                out[:, :, i, j] = np.max(X_slice, axis=(2, 3))
        
        return out
    
    def _relu(self, X):
        """ReLU激活函数"""
        return np.maximum(0, X)
    
    def _dropout(self, X, keep_prob=0.5):
        """Dropout正则化"""
        mask = np.random.binomial(1, keep_prob, size=X.shape) / keep_prob
        return X * mask
    
    def _softmax(self, X):
        """Softmax激活函数"""
        exp_scores = np.exp(X - np.max(X, axis=1, keepdims=True))
        return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    
    def compute_loss(self, y_pred, y, reg=0.001):
        """
        计算交叉熵损失和L2正则化
        
        参数:
            y_pred: 预测概率，形状 (N, num_classes)
            y: 真实标签，形状 (N,)
            reg: L2正则化强度
            
        返回:
            交叉熵损失 + L2正则化损失
        """
        m = y.shape[0]
        # 将y转换为one-hot编码
        y_one_hot = np.zeros((m, self.num_classes))
        y_one_hot[np.arange(m), y] = 1
        
        # 计算交叉熵损失
        loss = -np.sum(y_one_hot * np.log(y_pred + 1e-8)) / m
        
        # 添加L2正则化
        l2_loss = 0
        for i in range(1, 6):  # W1到W5
            if f'W{i}' in self.params:
                l2_loss += 0.5 * reg * np.sum(self.params[f'W{i}']**2)
        
        return loss + l2_loss
    
    def backward(self, y, reg=0.001):
        """
        反向传播
        
        参数:
            y: 真实标签，形状 (N,)
            reg: L2正则化强度
            
        返回:
            梯度字典
        """
        m = y.shape[0]
        grads = {}
        
        # 将y转换为one-hot编码
        y_one_hot = np.zeros((m, self.num_classes))
        y_one_hot[np.arange(m), y] = 1
        
        # 输出层梯度
        dA5 = self.activations['A5'] - y_one_hot
        
        # 全连接层2梯度
        grads['W5'] = np.dot(self.activations['A4'].T, dA5) / m + reg * self.params['W5']
        grads['b5'] = np.sum(dA5, axis=0) / m
        dZ4 = np.dot(dA5, self.params['W5'].T)
        
        # 全连接层1梯度(考虑dropout)
        dA4 = dZ4 * (self.activations['A4'] > 0)  # ReLU梯度
        
        # 批归一化梯度(简化处理)
        dBN4 = dA4
        
        # 全连接层1梯度
        grads['W4'] = np.dot(self.activations['F'].T, dBN4) / m + reg * self.params['W4']
        grads['b4'] = np.sum(dBN4, axis=0) / m
        
        # 卷积层梯度（简化处理）
        # 在实际实现中，需要详细处理卷积和池化的反向传播
        
        return grads
    
    def update_parameters(self, grads, learning_rate):
        """
        更新参数
        
        参数:
            grads: 梯度字典
            learning_rate: 学习率
        """
        # 更新全连接层参数
        for param_name in grads:
            self.params[param_name] -= learning_rate * grads[param_name]
    
    def predict(self, X):
        """
        预测函数
        
        参数:
            X: 输入数据，形状 (N, C, H, W)
            
        返回:
            预测类别，形状 (N,)
        """
        probs = self.forward(X, training=False)
        return np.argmax(probs, axis=1)

# MNIST数据集加载
class MNISTLoader:
    def __init__(self, data_dir='data'):
        self.data_dir = data_dir
        
        # MNIST文件
        self.train_img_file = os.path.join(data_dir, 'train-images-idx3-ubyte.gz')
        self.train_lbl_file = os.path.join(data_dir, 'train-labels-idx1-ubyte.gz')
        self.test_img_file = os.path.join(data_dir, 't10k-images-idx3-ubyte.gz')
        self.test_lbl_file = os.path.join(data_dir, 't10k-labels-idx1-ubyte.gz')
        
        # 下载地址
        self.url_base = 'http://yann.lecun.com/exdb/mnist/'
        self.train_img_url = self.url_base + 'train-images-idx3-ubyte.gz'
        self.train_lbl_url = self.url_base + 'train-labels-idx1-ubyte.gz'
        self.test_img_url = self.url_base + 't10k-images-idx3-ubyte.gz'
        self.test_lbl_url = self.url_base + 't10k-labels-idx1-ubyte.gz'
        
        # 确保数据目录存在
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
        
        # 下载所需文件
        self._download_mnist()
    
    def _download_mnist(self):
        """下载MNIST数据集"""
        # 检查并下载训练图像
        if not os.path.exists(self.train_img_file):
            print("下载训练图像...")
            self._download_file(self.train_img_url, self.train_img_file)
        
        # 检查并下载训练标签
        if not os.path.exists(self.train_lbl_file):
            print("下载训练标签...")
            self._download_file(self.train_lbl_url, self.train_lbl_file)
        
        # 检查并下载测试图像
        if not os.path.exists(self.test_img_file):
            print("下载测试图像...")
            self._download_file(self.test_img_url, self.test_img_file)
        
        # 检查并下载测试标签
        if not os.path.exists(self.test_lbl_file):
            print("下载测试标签...")
            self._download_file(self.test_lbl_url, self.test_lbl_file)
    
    def _download_file(self, url, file_path):
        """下载文件"""
        urllib.request.urlretrieve(url, file_path)
        print(f"已下载到 {file_path}")
    
    def load_mnist(self):
        """加载MNIST数据集"""
        # 加载训练数据
        with gzip.open(self.train_img_file, 'rb') as f:
            magic, num, rows, cols = struct.unpack('>IIII', f.read(16))
            train_imgs = np.frombuffer(f.read(), dtype=np.uint8).reshape(num, rows, cols)
        
        with gzip.open(self.train_lbl_file, 'rb') as f:
            magic, num = struct.unpack('>II', f.read(8))
            train_lbls = np.frombuffer(f.read(), dtype=np.uint8)
        
        # 加载测试数据
        with gzip.open(self.test_img_file, 'rb') as f:
            magic, num, rows, cols = struct.unpack('>IIII', f.read(16))
            test_imgs = np.frombuffer(f.read(), dtype=np.uint8).reshape(num, rows, cols)
        
        with gzip.open(self.test_lbl_file, 'rb') as f:
            magic, num = struct.unpack('>II', f.read(8))
            test_lbls = np.frombuffer(f.read(), dtype=np.uint8)
        
        # 规范化并重塑为CNN输入格式
        train_imgs = train_imgs.reshape(-1, 1, 28, 28).astype(np.float32) / 255.0
        test_imgs = test_imgs.reshape(-1, 1, 28, 28).astype(np.float32) / 255.0
        
        return (train_imgs, train_lbls), (test_imgs, test_lbls)

# 带学习率调度的训练函数
def train(model, train_data, train_labels, val_data, val_labels, test_data, test_labels, 
          batch_size=128, epochs=10, initial_lr=0.01, lr_decay=0.95, reg=0.001):
    """
    训练模型
    
    参数:
        model: 神经网络模型
        train_data: 训练数据，形状 (N, C, H, W)
        train_labels: 训练标签，形状 (N,)
        val_data: 验证数据，形状 (N, C, H, W)
        val_labels: 验证标签，形状 (N,)
        test_data: 测试数据，形状 (N, C, H, W)
        test_labels: 测试标签，形状 (N,)
        batch_size: 批量大小
        epochs: 训练轮数
        initial_lr: 初始学习率
        lr_decay: 学习率衰减因子
        reg: L2正则化强度
    """
    # 训练样本数
    m = train_data.shape[0]
    
    # 训练历史
    train_losses = []
    train_accuracies = []
    val_accuracies = []
    test_accuracies = []
    
    # 初始学习率
    learning_rate = initial_lr
    
    # 训练循环
    for epoch in range(epochs):
        start_time = time.time()
        epoch_loss = 0
        
        # 打乱训练数据
        indices = np.random.permutation(m)
        train_data_shuffled = train_data[indices]
        train_labels_shuffled = train_labels[indices]
        
        # 批量训练
        for i in range(0, m, batch_size):
            # 获取批量数据
            batch_data = train_data_shuffled[i:i+batch_size]
            batch_labels = train_labels_shuffled[i:i+batch_size]
            
            # 前向传播
            probs = model.forward(batch_data, training=True)
            
            # 计算损失
            loss = model.compute_loss(probs, batch_labels, reg=reg)
            epoch_loss += loss * len(batch_data)
            
            # 反向传播
            grads = model.backward(batch_labels, reg=reg)
            
            # 更新参数
            model.update_parameters(grads, learning_rate)
        
        # 计算平均损失
        epoch_loss /= m
        train_losses.append(epoch_loss)
        
        # 计算训练准确率
        train_accuracy = evaluate(model, train_data[:1000], train_labels[:1000])
        train_accuracies.append(train_accuracy)
        
        # 计算验证准确率
        val_accuracy = evaluate(model, val_data, val_labels)
        val_accuracies.append(val_accuracy)
        
        # 计算测试准确率
        test_accuracy = evaluate(model, test_data, test_labels)
        test_accuracies.append(test_accuracy)
        
        print(f"轮次 {epoch+1}/{epochs}, 损失: {epoch_loss:.4f}, 训练准确率: {train_accuracy:.4f}, 验证准确率: {val_accuracy:.4f}, 测试准确率: {test_accuracy:.4f}, 用时: {time.time()-start_time:.2f}s")
        
        # 学习率衰减
        learning_rate *= lr_decay
    
    return train_losses, train_accuracies, val_accuracies, test_accuracies

# 评估函数
def evaluate(model, data, labels):
    """
    评估模型性能
    
    参数:
        model: 神经网络模型
        data: 测试数据，形状 (N, C, H, W)
        labels: 测试标签，形状 (N,)
    
    返回:
        准确率
    """
    # 预测
    predictions = model.predict(data)
    
    # 计算准确率
    accuracy = np.mean(predictions == labels)
    
    return accuracy

# 绘制并保存准确率和损失曲线
def plot_metrics(train_losses, train_accuracies, val_accuracies, test_accuracies, save_dir='./results'):
    """
    绘制并保存训练过程中的指标变化曲线
    
    参数:
        train_losses: 训练损失列表
        train_accuracies: 训练准确率列表
        val_accuracies: 验证准确率列表
        test_accuracies: 测试准确率列表
        save_dir: 保存图表的目录
    """
    # 创建保存目录
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # 绘制损失曲线
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, 'b-o', label='train_losses', markersize=6)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title('train_losses')
    plt.legend()
    plt.grid(True)
    
    # 绘制准确率曲线 - 使用不同形状的标记点
    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, 'r-o', label='train_accuracies', markersize=6)
    plt.plot(val_accuracies, 'g-s', label='val_accuracies', markersize=6)  # 方形标记
    plt.plot(test_accuracies, 'm-^', label='test_accuracies', markersize=6)  # 三角形标记
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.title('accuracy')
    plt.legend()
    plt.grid(True)
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图表
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    save_path = os.path.join(save_dir, f'training_metrics_{timestamp}.png')
    plt.savefig(save_path, dpi=300)
    print(f"指标曲线已保存至: {save_path}")
    
    # 显示图表
    plt.show()

# 可视化预测结果
def visualize_predictions(model, images, labels, num_samples=5):
    """
    随机选择几张图像并展示它们的预测结果
    
    参数:
        model: 训练好的模型
        images: 图像数据，形状 (N, C, H, W)
        labels: 真实标签，形状 (N,)
        num_samples: 要展示的样本数量
    """
    # 类别标签映射
    class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    
    # 随机选择样本
    indices = np.random.choice(len(images), num_samples, replace=False)
    
    # 获取预测结果
    batch_images = images[indices]
    batch_labels = labels[indices]
    predictions = model.predict(batch_images)
    
    # 创建图表
    fig, axes = plt.subplots(1, num_samples, figsize=(15, 3))
    
    # 展示每个样本
    for i in range(num_samples):
        # 获取图像并转换为合适的显示格式
        img = batch_images[i, 0]  # 第一个通道，MNIST只有一个通道
        
        # 显示图像
        axes[i].imshow(img, cmap='gray')
        
        # 获取真实标签和预测标签
        true_label = batch_labels[i]
        pred_label = predictions[i]
        
        # 设置标题：绿色表示预测正确，红色表示预测错误
        title_color = 'green' if true_label == pred_label else 'red'
        axes[i].set_title(f'真实: {class_names[true_label]}\n预测: {class_names[pred_label]}', 
                         color=title_color)
        
        # 移除坐标轴
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()

# 主函数
def main():
    # 加载MNIST数据集
    mnist_loader = MNISTLoader()
    (train_imgs, train_lbls), (test_imgs, test_lbls) = mnist_loader.load_mnist()
    
    # 分割训练集为训练集和验证集
    train_size = 50000
    val_size = 10000
    train_data, train_labels = train_imgs[:train_size], train_lbls[:train_size]
    val_data, val_labels = train_imgs[train_size:train_size+val_size], train_lbls[train_size:train_size+val_size]
    
    # 创建AlexNet模型
    model = AlexNet(input_shape=(1, 28, 28), num_classes=10)
    
    # 训练模型
    print("开始训练...")
    train_losses, train_accuracies, val_accuracies, test_accuracies = train(
        model=model,
        train_data=train_data,
        train_labels=train_labels,
        val_data=val_data,
        val_labels=val_labels,
        test_data=test_imgs,
        test_labels=test_lbls,
        batch_size=256,
        epochs=10,
        initial_lr=0.01,    #初始学习率
        lr_decay=0.95,      #学习率衰减因子
        reg=0.001           #L2正则化强度
    )
    
    # 在完整测试集上评估
    test_accuracy = evaluate(model, test_imgs, test_lbls)
    print(f"最终测试集准确率: {test_accuracy:.4f}")
    
    # 绘制并保存指标曲线
    plot_metrics(train_losses, train_accuracies, val_accuracies, test_accuracies)
    
    # 可视化一些随机预测结果
    print("随机展示预测结果...")
    visualize_predictions(model, test_imgs, test_lbls, num_samples=8)

if __name__ == "__main__":
    main()