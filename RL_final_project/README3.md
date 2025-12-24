# Snake AI - 深度强化学习贪吃蛇项目

基于深度Q网络(DQN)的智能贪吃蛇游戏AI，支持多平台GPU加速。

## 🚀 功能特点

- 🧠 **深度强化学习** - 使用CNN+DQN算法训练智能体
- ⚡ **多平台GPU加速** - 支持NVIDIA CUDA、Apple MPS和CPU
- 🎮 **多种游戏模式** - AI自动游戏 + 玩家手动试玩
- 📊 **实时训练监控** - 可视化训练进度和性能指标
- 💾 **智能模型管理** - 自动保存和加载最佳模型

## 📋 系统要求

### 硬件要求
- **最低配置**: 4GB RAM, 支持Python 3.8+
- **推荐配置**: 8GB+ RAM, 支持CUDA的GPU或Apple Silicon芯片

### 软件要求
- Python 3.8 或更高版本
- 支持的操作系统: Windows 10+, macOS 10.15+, Ubuntu 18.04+

## 🛠️ 安装指南

### 1. 克隆项目
```bash
git clone hhttps://github.com/Leo0902110/snake_cnn_ai
cd snake-ai
```


### 2. 创建Python虚拟环境
```bash
# Windows
python -m venv snake_env
snake_env\Scripts\activate

# macOS/Linux
python3 -m venv snake_env
source snake_env/bin/activate
```

### 3. 安装依赖
```bash
pip install -r requirements.txt
```




### PyTorch安装说明（可选）
如需特定版本的PyTorch，请访问 https://pytorch.org/get-started/locally/ 获取适合您系统的安装命令。

## 🎯 使用方法

### 训练AI模型
```bash
python snake_ai.py train
```
**训练参数**:
- 目标轮数: 20,000 episodes
- 每轮游戏数: 20局
- 自动保存间隔: 每250轮
- 预计时间: 2-8小时（取决于硬件）

### 观看AI玩游戏
```bash
python snake_ai.py play
```

### 玩家试玩模式
```bash
python snake_ai.py human
```

### 交互式选择模式
```bash
python snake_ai.py
```
然后根据提示选择训练或游戏模式。

## 🎮 游戏控制

### AI模式
- 自动运行，无需人工干预
- 按关闭按钮或Ctrl+C退出

###玩家模式
- **方向键**控制蛇的移动
- **↑↓←→**: 上下左右移动
- 游戏结束后自动重新开始

## 📊 训练监控

训练过程中会实时显示以下指标：
- 🍎 **苹果消耗进度** - 每轮平均吃到的苹果数
- 🏆 **奖励趋势** - 强化学习奖励变化
- 📏 **蛇长度进展** - 蛇的平均和最大长度
- 📉 **训练损失** - 模型学习损失（对数坐标）

## 📁 项目结构

```
snake_ai/
├── snake_ai.py              # 主程序文件
├── requirements.txt          # 依赖包列表
├── README.md                # 项目说明文档
├── models/                  # 模型保存目录（自动创建）
│   ├── snake_model_*.pth    # 训练过程中的模型检查点
│   ├── snake_model_final.pth # 最终训练模型
│   └── model_performance.json # 模型性能记录
├── training_data.npz        # 训练数据文件
├── training_results.png     # 训练结果图表
└── training_summary.png     # 训练总结图表
```

## ⚙️ 配置参数

在代码中可以调整以下关键参数：

```python
# 游戏设置
GRIDSIZE = 15           # 游戏网格大小 (建议: 10-20)

# 训练参数
NUM_EPISODES = 20000    # 训练总轮数
BATCH_SIZE = 256        # 经验回放批次大小
LR = 1e-4               # 学习率
GAMMA = 0.99            # 折扣因子

# 性能设置
GAMES_IN_EPISODE = 20   # 每轮游戏局数
TARGET_UPDATE = 10      # 目标网络更新频率
```

## 🐛 故障排除

### 常见问题

**Q: 训练速度太慢**
A: 确保已启用GPU加速，检查CUDA/MPS是否正常工作

**Q: 内存不足错误**
A: 减小`BATCH_SIZE`或`GRIDSIZE`参数

**Q: 无法显示游戏窗口**
A: 确保已安装pygame且系统支持图形界面

**Q: 模型无法加载**
A: 检查models目录是否存在有效的模型文件

### 性能优化建议

1. **GPU加速**: 使用支持CUDA的NVIDIA显卡或Apple Silicon芯片
2. **批量大小**: 根据显存调整`BATCH_SIZE`
3. **网格大小**: 较小的网格可加快训练速度
4. **游戏数量**: 减少`GAMES_IN_EPISODE`可加快每轮训练

## 📈 预期结果

经过完整训练后，AI应该能够：
- ✅ 稳定获得30+个苹果
- ✅ 避免撞墙和自撞
- ✅ 有效寻找苹果路径
- ✅ 适应不同游戏局面

## 🤝 贡献指南

欢迎提交Issue和Pull Request来改进项目！

### 开发环境设置
```bash
# 1. Fork项目
# 2. 克隆你的fork
git clone https://github.com/Leo0902110/snake_cnn_ai
cd snake-ai

# 3. 创建功能分支
git checkout -b feature/amazing-feature

# 4. 提交更改
git commit -m 'Add some amazing feature'

# 5. 推送到分支
git push origin feature/amazing-feature

# 6. 创建Pull Request
```

## 📄 许可证

本项目采用MIT许可证 - 详见 LICENSE 文件。

## 🙏 致谢

- 感谢PyTorch团队提供的优秀深度学习框架
- 感谢开源社区的各种工具和库
- 灵感来源于经典的强化学习教程和项目

---

**Happy Coding! 🐍✨**