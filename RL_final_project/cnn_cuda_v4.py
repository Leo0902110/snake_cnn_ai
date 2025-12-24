import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import time
from collections import deque
import matplotlib

# 设置matplotlib后端以确保显示
try:
    matplotlib.use('TkAgg')  # 尝试TkAgg后端
except:
    try:
        matplotlib.use('Qt5Agg')  # 尝试Qt5后端
    except:
        matplotlib.use('Agg')  # 如果都不行，使用非交互式后端
import matplotlib.pyplot as plt
import os
import sys
import json

# ==========================================
# Part 0: Device Configuration (Added MPS Support)
# ==========================================
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    print(f">>> 成功启用 NVIDIA GPU 加速 (CUDA) - 设备: {torch.cuda.get_device_name()} <<<")
    print(f">>> CUDA版本: {torch.version.cuda} <<<")
# --- 新增: Apple Silicon (M1/M2/M3) MPS 支持 ---
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
    print(f">>> 成功启用 Apple Metal 加速 (MPS) - 适用于 Mac M1/M2/M3 <<<")
else:
    DEVICE = torch.device("cpu")
    print(">>> 未检测到 GPU，使用 CPU 模式 <<<")

# ==========================================
# Part 1: Game Logic (保持不变)
# ==========================================

player_moves = {
    'L': np.array([-1, 0]),
    'R': np.array([1, 0]),
    'U': np.array([0, -1]),
    'D': np.array([0, 1])
}
initial_playersize = 4


class snakeclass(object):
    def __init__(self, gridsize):
        self.gridsize = gridsize
        self.reset()

    def reset(self):
        head_x, head_y = self.gridsize // 2, self.gridsize // 2
        self.pos = np.array([head_x, head_y]).astype('int')
        self.dir = np.array([1, 0])
        self.len = initial_playersize
        self.prevpos = []
        for i in range(self.len - 1, -1, -1):
            self.prevpos.append(np.array([head_x - i, head_y]).astype('int'))

    def move(self):
        self.pos += self.dir
        self.prevpos.append(self.pos.copy())
        self.prevpos = self.prevpos[-self.len - 1:]

    def checkdead(self, pos):
        if pos[0] <= -1 or pos[0] >= self.gridsize:
            return True
        elif pos[1] <= -1 or pos[1] >= self.gridsize:
            return True
        if len(self.prevpos) > 1:
            body = np.array(self.prevpos[:-1])
            if (body == pos).all(axis=1).any():
                return True
        return False

    def __len__(self):
        return self.len + 1


class appleclass(object):
    def __init__(self, gridsize):
        self.gridsize = gridsize
        self.pos = np.zeros(2, dtype='int')

    def eaten(self, snake):
        all_grids = set((x, y) for x in range(self.gridsize) for y in range(self.gridsize))
        snake_body = set((p[0], p[1]) for p in snake.prevpos)
        available_grids = list(all_grids - snake_body)
        if not available_grids:
            return False
        idx = np.random.choice(len(available_grids))
        self.pos = np.array(available_grids[idx])
        return True


class GameEnvironment(object):
    def __init__(self, gridsize, nothing, dead, apple):
        self.snake = snakeclass(gridsize)
        self.apple = appleclass(gridsize)
        self.game_over = False
        self.gridsize = gridsize
        self.reward_nothing = nothing
        self.reward_dead = dead
        self.reward_apple = apple
        self.time_since_apple = 0
        self.apples_eaten = 0

    def resetgame(self):
        self.snake.reset()
        self.apple.eaten(self.snake)
        self.game_over = False
        self.time_since_apple = 0
        self.apples_eaten = 0

    def update_boardstate(self, move):
        reward = self.reward_nothing
        Done = False
        prev_dist = np.linalg.norm(self.snake.pos - self.apple.pos)

        invalid_move = False
        # Move Mapping: 0:L, 1:R, 2:U, 3:D
        if move == 0 and (self.snake.dir == player_moves['R']).all():
            invalid_move = True
        elif move == 1 and (self.snake.dir == player_moves['L']).all():
            invalid_move = True
        elif move == 2 and (self.snake.dir == player_moves['D']).all():
            invalid_move = True
        elif move == 3 and (self.snake.dir == player_moves['U']).all():
            invalid_move = True

        if move == 0:
            if not (self.snake.dir == player_moves['R']).all(): self.snake.dir = player_moves['L']
        if move == 1:
            if not (self.snake.dir == player_moves['L']).all(): self.snake.dir = player_moves['R']
        if move == 2:
            if not (self.snake.dir == player_moves['D']).all(): self.snake.dir = player_moves['U']
        if move == 3:
            if not (self.snake.dir == player_moves['U']).all(): self.snake.dir = player_moves['D']

        self.snake.move()

        if self.snake.checkdead(self.snake.pos):
            self.game_over = True
            reward = self.reward_dead
            Done = True
            return reward, Done, self.apples_eaten

        if (self.snake.pos == self.apple.pos).all():
            success = self.apple.eaten(self.snake)
            if not success:
                self.game_over = True
                Done = True
                reward = self.reward_apple * 2
                return reward, Done, self.apples_eaten
            self.snake.len += 1
            self.time_since_apple = 0
            self.apples_eaten += 1
            reward = self.reward_apple
            return reward, Done, self.apples_eaten

        self.time_since_apple += 1
        if self.time_since_apple >= 500:
            self.game_over = True
            reward = self.reward_dead
            Done = True
            return reward, Done, self.apples_eaten

        curr_dist = np.linalg.norm(self.snake.pos - self.apple.pos)
        if invalid_move:
            reward -= 0.1
        elif curr_dist < prev_dist:
            reward += 0.05
        else:
            reward -= 0.05

        return reward, Done, self.apples_eaten


# ==========================================
# Part 2: CNN Model Logic
# ==========================================

class CNN_QNet(nn.Module):
    def __init__(self, grid_w, grid_h, output_size):
        super(CNN_QNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.fc_input_dims = 64 * grid_w * grid_h
        self.fc1 = nn.Linear(self.fc_input_dims, 512)
        self.fc2 = nn.Linear(512, output_size)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(-1, self.fc_input_dims)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def get_board_state_image(env):
    state = np.zeros((3, env.gridsize, env.gridsize), dtype=np.float32)
    if len(env.snake.prevpos) > 1:
        pts = np.array(env.snake.prevpos[:-1])
        valid_mask = (pts[:, 0] >= 0) & (pts[:, 0] < env.gridsize) & \
                     (pts[:, 1] >= 0) & (pts[:, 1] < env.gridsize)
        valid_pts = pts[valid_mask]
        if len(valid_pts) > 0:
            state[1, valid_pts[:, 1], valid_pts[:, 0]] = 1.0
    head = env.snake.pos
    if 0 <= head[0] < env.gridsize and 0 <= head[1] < env.gridsize:
        state[0, head[1], head[0]] = 1.0
    apple = env.apple.pos
    if 0 <= apple[0] < env.gridsize and 0 <= apple[1] < env.gridsize:
        state[2, apple[1], apple[0]] = 1.0
    return state


# ==========================================
# Part 3: Replay Buffer
# ==========================================

class ReplayMemory(object):
    def __init__(self, max_size):
        self.max_size = max_size
        self.buffer = deque(maxlen=max_size)

    def push(self, state, action, reward, next_state, done):
        experience = (state, action, reward, next_state, done)
        self.buffer.append(experience)

    def sample(self, batch_size):
        sample_size = min(len(self.buffer), batch_size)
        batch = random.sample(self.buffer, sample_size)
        state, action, reward, next_state, done = zip(*batch)
        return np.array(state), np.array(action), np.array(reward), np.array(next_state), np.array(done)

    def truncate(self):
        pass

    def __len__(self):
        return len(self.buffer)


# ==========================================
# Part 4: Training Parameters
# ==========================================

gridsize = 15
policy_net = CNN_QNet(grid_w=gridsize, grid_h=gridsize, output_size=4).to(DEVICE)
target_net = CNN_QNet(grid_w=gridsize, grid_h=gridsize, output_size=4).to(DEVICE)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

# 只有在 CUDA 模式下才开启 cudnn benchmark，避免 MPS 报错
if DEVICE.type == 'cuda':
    torch.backends.cudnn.benchmark = True

epsilon = 1.0
epsilon_min = 0.01
epsilon_decay = 0.9995
GAMMA = 0.99
lr = 1e-4
TARGET_UPDATE = 10

board = GameEnvironment(gridsize, nothing=0, dead=-1, apple=1)
memory = ReplayMemory(100000)
optimizer = torch.optim.Adam(policy_net.parameters(), lr=lr)
MSE = nn.MSELoss()

num_episodes = 20000
num_updates = 50
print_every = 10
games_in_episode = 20
batch_size = 256


# ==========================================
# Part 5: Training Functions with Real-time Plotting
# ==========================================

def run_episode(num_games):
    global epsilon
    run = True
    games_played = 0
    total_reward = 0
    total_apples = 0
    len_array = []
    current_state = get_board_state_image(board)

    while run:
        rand = np.random.uniform(0, 1)
        if rand > epsilon:
            state_tensor = torch.tensor(current_state, dtype=torch.float32).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                action_0 = policy_net(state_tensor)
                action = torch.argmax(action_0).item()
        else:
            action = np.random.randint(0, 4)

        reward, done, apples_count = board.update_boardstate(action)
        next_state = get_board_state_image(board)
        memory.push(current_state, action, reward, next_state, done)
        total_reward += reward
        current_state = next_state

        if board.game_over == True:
            games_played += 1
            len_array.append(board.snake.len)
            total_apples += apples_count
            board.resetgame()
            current_state = get_board_state_image(board)
            if num_games == games_played:
                run = False

    if epsilon > epsilon_min:
        epsilon *= epsilon_decay

    avg_len_of_snake = np.mean(len_array)
    max_len_of_snake = np.max(len_array)
    avg_reward = total_reward / num_games
    avg_apples = total_apples / num_games
    return avg_reward, avg_apples, avg_len_of_snake, max_len_of_snake


def learn(num_updates, batch_size):
    if len(memory) < batch_size:
        return 0.0
    total_loss = 0
    for i in range(num_updates):
        optimizer.zero_grad()
        states, actions, rewards, next_states, dones = memory.sample(batch_size)
        states = torch.as_tensor(states, dtype=torch.float32).to(DEVICE)
        actions = torch.as_tensor(actions, dtype=torch.long).to(DEVICE)
        rewards = torch.as_tensor(rewards, dtype=torch.float32).to(DEVICE)
        next_states = torch.as_tensor(next_states, dtype=torch.float32).to(DEVICE)
        dones = torch.as_tensor(dones, dtype=torch.float32).to(DEVICE)
        q_local = policy_net(states)
        Q_expected = q_local.gather(1, actions.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            next_q_value = target_net(next_states)
            Q_targets_next = torch.max(next_q_value, 1)[0] * (1 - dones)
            Q_targets = rewards + GAMMA * Q_targets_next
        loss = MSE(Q_expected, Q_targets)
        total_loss += loss.item()
        loss.backward()
        
        # Clip gradients
        torch.nn.utils.clip_grad_norm_(policy_net.parameters(), max_norm=1.0)
        optimizer.step()
    return total_loss / num_updates


def train():
    scores_deque = deque(maxlen=100)
    scores_array = []
    rewards_array = []
    avg_scores_array = []
    avg_len_array = []
    avg_max_len_array = []
    loss_array = []
    model_performance = {}  # 记录模型性能

    time_start = time.time()

    if not os.path.exists('./dir_chk_len_cnn'):
        os.makedirs('./dir_chk_len_cnn')

    device_name = "CUDA" if DEVICE.type == 'cuda' else ("MPS" if DEVICE.type == 'mps' else "CPU")
    print(f"开始训练 (CNN + Target Net + {device_name})... 目标: {num_episodes} Episodes")

    # 初始化实时图表
    plt.ion()
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    plt.subplots_adjust(wspace=0.3, hspace=0.3)
    fig.suptitle(f'Snake AI Training Progress ({device_name})', fontsize=16)

    for i_episode in range(num_episodes + 1):
        if i_episode % 100 == 0 and DEVICE.type == 'cuda':
            torch.cuda.empty_cache()
        elif i_episode % 100 == 0 and DEVICE.type == 'mps':
             torch.mps.empty_cache() # MPS 清理缓存

        avg_reward, avg_apples, avg_len, max_len = run_episode(games_in_episode)
        scores_deque.append(avg_apples)
        scores_array.append(avg_apples)
        rewards_array.append(avg_reward)
        avg_len_array.append(avg_len)
        avg_max_len_array.append(max_len)
        avg_score = np.mean(scores_deque)
        avg_scores_array.append(avg_score)
        avg_loss = learn(num_updates, batch_size)
        loss_array.append(avg_loss)

        if i_episode % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())

        # 实时更新图表
        if i_episode % print_every == 0 and i_episode > 0:
            episodes = np.arange(1, len(scores_array) + 1)

            ax1.clear()
            ax2.clear()
            ax3.clear()
            ax4.clear()

            # 苹果数量变化
            ax1.plot(episodes, scores_array, 'b-', alpha=0.7, label='Apples per Episode')
            ax1.plot(episodes, avg_scores_array, 'r-', linewidth=2, label='Moving Avg (100 episodes)')
            ax1.set_xlabel('Training Episodes')
            ax1.set_ylabel('Apples Collected')
            ax1.set_title('Apple Consumption Progress')
            ax1.legend()
            ax1.grid(True, alpha=0.3)

            # 奖励变化
            ax2.plot(episodes, rewards_array, 'g-', alpha=0.7)
            ax2.set_xlabel('Training Episodes')
            ax2.set_ylabel('Reward')
            ax2.set_title('Reward Trend')
            ax2.grid(True, alpha=0.3)

            # 蛇长度变化
            ax3.plot(episodes, avg_len_array, 'orange', label='Average Length')
            ax3.plot(episodes, avg_max_len_array, 'red', label='Max Length')
            ax3.set_xlabel('Training Episodes')
            ax3.set_ylabel('Snake Length')
            ax3.set_title('Snake Length Progression')
            ax3.legend()
            ax3.grid(True, alpha=0.3)

            # 损失变化
            ax4.plot(episodes, loss_array, 'purple')
            ax4.set_xlabel('Training Episodes')
            ax4.set_ylabel('Loss')
            ax4.set_title('Training Loss (Log Scale)')
            ax4.set_yscale('log')
            ax4.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.draw()
            plt.pause(0.01)

        # 保存模型和性能记录
        if i_episode % 250 == 0 and i_episode > 0:
            torch.save(policy_net.state_dict(), f'./dir_chk_len_cnn/Snake_CNN_{i_episode}')
            # 记录模型性能
            model_performance[i_episode] = {
                'avg_apples': avg_apples,
                'avg_reward': avg_reward,
                'avg_len': avg_len,
                'max_len': max_len,
                'avg_score': avg_score
            }
            # 保存性能记录
            with open('./dir_chk_len_cnn/model_performance.json', 'w') as f:
                json.dump(model_performance, f)

        # 控制台输出
        if i_episode % print_every == 0 and i_episode > 0:
            dt = (int)(time.time() - time_start)
            gpu_mem = 0
            if DEVICE.type == 'cuda':
                gpu_mem = torch.cuda.memory_allocated() / 1024 ** 3
            elif DEVICE.type == 'mps':
                gpu_mem = torch.mps.current_allocated_memory() / 1024 ** 3

            print(f'Ep: {i_episode:6}, Loss: {avg_loss:.4f}, Apples: {avg_apples:.2f}, '
                  f'Reward: {avg_reward:.2f}, Max.Len: {max_len:.2f}, '
                  f'Eps: {epsilon:.3f}, GPU: {gpu_mem:.1f}GB, Time: {dt // 3600:02}:{dt % 3600 // 60:02}:{dt % 60:02}')

        memory.truncate()

    # 训练结束后的处理
    plt.ioff()

    # 保存最终图表
    plt.savefig('./training_results_final.png', dpi=300, bbox_inches='tight')

    # 保存训练数据
    np.savez('./training_data.npz',
             scores=scores_array,
             avg_scores=avg_scores_array,
             rewards=rewards_array,
             lengths=avg_len_array,
             max_lengths=avg_max_len_array,
             losses=loss_array)

    print("训练完成！")
    return scores_array, avg_scores_array, avg_len_array, avg_max_len_array


# ==========================================
# Part 6: Play Function & Human Mode
# ==========================================

def find_best_model():
    """自动选择表现最好的模型"""
    model_dir = './dir_chk_len_cnn'
    performance_file = os.path.join(model_dir, 'model_performance.json')

    if not os.path.exists(performance_file):
        print("未找到模型性能记录文件，将使用最新模型")
        return None

    try:
        with open(performance_file, 'r') as f:
            model_performance = json.load(f)

        if not model_performance:
            print("模型性能记录为空，将使用最新模型")
            return None

        # 找到平均苹果数最高的模型
        best_epoch = max(model_performance.keys(), key=lambda x: model_performance[x]['avg_apples'])
        best_score = model_performance[best_epoch]['avg_apples']
        print(f"找到最佳模型: Episode {best_epoch}, 平均苹果数: {best_score:.2f}")
        return int(best_epoch)

    except Exception as e:
        print(f"读取模型性能记录失败: {e}，将使用最新模型")
        return None

def play_ai():
    import pygame

    # 自动选择最佳模型
    best_epoch = find_best_model()
    model_dir = './dir_chk_len_cnn'

    if not os.path.exists(model_dir):
        print(f"错误：找不到模型文件夹 {model_dir}")
        return

    files = [f for f in os.listdir(model_dir) if f.startswith('Snake_CNN_')]
    if not files:
        print("错误：文件夹中没有找到模型文件！")
        return

    try:
        if best_epoch is not None:
            # 尝试加载最佳模型
            model_filename = f"Snake_CNN_{best_epoch}"
            if model_filename in files:
                print(f"正在加载最佳模型: {model_filename}")
            else:
                files = sorted(files, key=lambda x: int(x.split('_')[2]))
                model_filename = files[-1]
                print(f"最佳模型文件不存在，改为加载最新模型: {model_filename}")
        else:
            files = sorted(files, key=lambda x: int(x.split('_')[2]))
            model_filename = files[-1]
            print(f"正在加载最新模型: {model_filename}")

    except Exception as e:
        print(f"文件名解析错误: {e}，默认使用第一个文件")
        model_filename = files[0]

    model_path = os.path.join(model_dir, model_filename)
    
    # 播放时使用CPU，避免不必要的显存占用
    play_device = torch.device("cpu")
    model = CNN_QNet(grid_w=gridsize, grid_h=gridsize, output_size=4)
    try:
        model.load_state_dict(torch.load(model_path, map_location=play_device))
        model.to(play_device)
        model.eval()
    except Exception as e:
        print(f"模型加载失败: {e}")
        return

    pygame.init()
    BLOCK_SIZE = 30
    GRID_W = gridsize
    GRID_H = gridsize
    WINDOW_W = GRID_W * BLOCK_SIZE
    WINDOW_H = GRID_H * BLOCK_SIZE

    display = pygame.display.set_mode((WINDOW_W, WINDOW_H))
    pygame.display.set_caption(f'Snake AI (Bot) - {model_filename}')
    clock = pygame.time.Clock()
    font = pygame.font.SysFont('arial', 25)

    env = GameEnvironment(gridsize=GRID_W, nothing=0, dead=-1, apple=1)
    env.resetgame()

    total_score = 0
    running = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                pygame.quit()
                return

        if not env.game_over:
            state_numpy = get_board_state_image(env)
            state_tensor = torch.tensor(state_numpy, dtype=torch.float32).unsqueeze(0).to(play_device)

            with torch.no_grad():
                q_values = model(state_tensor)
                action = torch.argmax(q_values).item()

            reward, done, apples = env.update_boardstate(action)
            total_score = apples

        display.fill((0, 0, 0))

        # 绘制蛇身
        for i, pos in enumerate(env.snake.prevpos):
            color = (0, 255, 0) if i == len(env.snake.prevpos) - 1 else (0, 200, 0)
            pygame.draw.rect(display, color,
                             pygame.Rect(pos[0] * BLOCK_SIZE, pos[1] * BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE))

        # 绘制苹果
        apple = env.apple.pos
        pygame.draw.rect(display, (255, 0, 0),
                         pygame.Rect(apple[0] * BLOCK_SIZE, apple[1] * BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE))

        # 显示分数
        text = font.render(f"Apples: {total_score}", True, (255, 255, 255))
        display.blit(text, [0, 0])

        if env.game_over:
            game_over_text = font.render("GAME OVER", True, (255, 0, 0))
            text_rect = game_over_text.get_rect(center=(WINDOW_W / 2, WINDOW_H / 2))
            display.blit(game_over_text, text_rect)
            pygame.display.flip()
            print(f"AI 游戏结束！得分: {total_score}")
            pygame.time.wait(2000)
            break

        pygame.display.flip()
        clock.tick(15) # AI 速度

    pygame.quit()

# --- 新增: 人类试玩模式 ---
def human_play():
    import pygame
    
    pygame.init()
    BLOCK_SIZE = 30
    GRID_W = gridsize
    GRID_H = gridsize
    WINDOW_W = GRID_W * BLOCK_SIZE
    WINDOW_H = GRID_H * BLOCK_SIZE

    display = pygame.display.set_mode((WINDOW_W, WINDOW_H))
    pygame.display.set_caption('Snake - Human Play Mode (Wait for Start)')
    clock = pygame.time.Clock()
    font = pygame.font.SysFont('arial', 25)
    big_font = pygame.font.SysFont('arial', 40)

    env = GameEnvironment(gridsize=GRID_W, nothing=0, dead=-1, apple=1)
    env.resetgame()

    # --- 新增：准备阶段 ---
    waiting = True
    while waiting:
        display.fill((0, 0, 0))
        # 显示提示信息
        msg = font.render("PRESS ANY KEY TO START", True, (255, 255, 255))
        msg_rect = msg.get_rect(center=(WINDOW_W // 2, WINDOW_H // 2))
        display.blit(msg, msg_rect)
        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return
            if event.type == pygame.KEYDOWN:
                waiting = False

    # --- 新增：倒计时阶段 (3, 2, 1) ---
    for i in range(3, 0, -1):
        display.fill((0, 0, 0))
        count_text = big_font.render(str(i), True, (255, 255, 0))
        count_rect = count_text.get_rect(center=(WINDOW_W // 2, WINDOW_H // 2))
        display.blit(count_text, count_rect)
        pygame.display.flip()
        pygame.time.wait(1000) # 等待1秒

    # --- 游戏正式开始 ---
    total_score = 0
    running = True
    current_action = 1 # 初始向右
    
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                return
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT: current_action = 0
                elif event.key == pygame.K_RIGHT: current_action = 1
                elif event.key == pygame.K_UP: current_action = 2
                elif event.key == pygame.K_DOWN: current_action = 3

        if not env.game_over:
            reward, done, apples = env.update_boardstate(current_action)
            total_score = apples

        display.fill((0, 0, 0))
        # 绘制蛇和苹果 (逻辑同前)
        for i, pos in enumerate(env.snake.prevpos):
            color = (0, 255, 0) if i == len(env.snake.prevpos) - 1 else (0, 200, 0)
            pygame.draw.rect(display, color, pygame.Rect(pos[0]*BLOCK_SIZE, pos[1]*BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE))
        
        apple = env.apple.pos
        pygame.draw.rect(display, (255, 0, 0), pygame.Rect(apple[0]*BLOCK_SIZE, apple[1]*BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE))
        
        text = font.render(f"Score: {total_score}", True, (255, 255, 255))
        display.blit(text, [10, 10])

        if env.game_over:
            # 死亡后也停留一下，不要立即关掉窗口
            over_text = big_font.render("GAME OVER", True, (255, 0, 0))
            display.blit(over_text, over_text.get_rect(center=(WINDOW_W//2, WINDOW_H//2)))
            pygame.display.flip()
            pygame.time.wait(2000)
            break

        pygame.display.flip()
        clock.tick(5) # 速度调为10，给人类反应时间

    pygame.quit()

# ==========================================
# Part 7: Main Program
# ==========================================

if __name__ == '__main__':
    print("命令行参数:", sys.argv)
    print("用法:")
    print("  python script.py         -> 训练模式")
    print("  python script.py play    -> 观看AI玩")
    print("  python script.py human   -> 人类试玩")

    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()
        
        if mode == 'play':
            while True:
                play_ai()
                cmd = input("按 Enter 再来一局 AI 演示，输入 'q' 退出: ")
                if cmd.lower() == 'q':
                    break
        
        elif mode == 'human':
            while True:
                human_play()
                cmd = input("按 Enter 再试一次，输入 'q' 退出: ")
                if cmd.lower() == 'q':
                    break
        else:
            print(f"未知参数: {mode}，默认进入训练模式...")
            train()
    else:
        # 默认训练模式
        scores, avg_scores, avg_len_of_snake, max_len_of_snake = train()

        # 训练结束后显示最终图表
        fig = plt.figure(figsize=(12, 5))
        ax = fig.add_subplot(121)
        plt.plot(np.arange(1, len(scores) + 1), scores, label="Apples Eaten")
        plt.plot(np.arange(1, len(avg_scores) + 1), avg_scores, label="Avg Apples (100 eps)")
        plt.legend(bbox_to_anchor=(1.05, 1))
        plt.ylabel('Apples')
        plt.xlabel('Episodes #')

        ax1 = fig.add_subplot(122)
        plt.plot(np.arange(1, len(avg_len_of_snake) + 1), avg_len_of_snake, label="Avg Len")
        plt.legend(bbox_to_anchor=(1.05, 1))
        plt.ylabel('Snake Length')
        plt.xlabel('Episodes #')

        plt.tight_layout()
        plt.savefig('./training_summary.png', dpi=300, bbox_inches='tight')
        plt.show()

        print("训练总结图表已保存为: ./training_summary.png")