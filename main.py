import pygame
import numpy as np
import random
import sys

pygame.init()

WINDOW_SIZE = 600
GRID_SIZE = 3
CELL_SIZE = WINDOW_SIZE // GRID_SIZE

WHITE = (255, 255, 255)
LINE_COLOR = (0, 0, 0)
X_COLOR = (242, 85, 96)
O_COLOR = (28, 170, 156)
BG_COLOR = (255, 255, 255)

# 字体设置
FONT = pygame.font.Font(None, 100)

# 创建 Pygame 窗口
screen = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE))
pygame.display.set_caption("Tic-Tac-Toe")

# 绘制棋盘
def draw_board():
    screen.fill(BG_COLOR)
    # 绘制横线和竖线
    for i in range(1, GRID_SIZE):
        pygame.draw.line(screen, LINE_COLOR, (0, i * CELL_SIZE), (WINDOW_SIZE, i * CELL_SIZE), 5)
        pygame.draw.line(screen, LINE_COLOR, (i * CELL_SIZE, 0), (i * CELL_SIZE, WINDOW_SIZE), 5)

# 绘制 X 和 O
def draw_marks(board):
    for row in range(GRID_SIZE):
        for col in range(GRID_SIZE):
            if board[row, col] == 1:
                draw_x(row, col)
            elif board[row, col] == -1:
                draw_o(row, col)  # Ensure this function is being called correctly

# 绘制 X
def draw_x(row, col):
    pygame.draw.line(screen, X_COLOR, (col * CELL_SIZE + 15, row * CELL_SIZE + 15),
                     ((col + 1) * CELL_SIZE - 15, (row + 1) * CELL_SIZE - 15), 15)
    pygame.draw.line(screen, X_COLOR, ((col + 1) * CELL_SIZE - 15, row * CELL_SIZE + 15),
                     (col * CELL_SIZE + 15, (row + 1) * CELL_SIZE - 15), 15)

# 绘制 O
def draw_o(row, col):
    pygame.draw.circle(screen, O_COLOR, (col * CELL_SIZE + CELL_SIZE // 2, row * CELL_SIZE + CELL_SIZE // 2),
                       CELL_SIZE // 2 - 15, 15)

# 检查是否有人赢得游戏
def check_winner(board):
    # 检查行
    for row in range(GRID_SIZE):
        if abs(sum(board[row, :])) == GRID_SIZE:
            return board[row, 0]
    # 检查列
    for col in range(GRID_SIZE):
        if abs(sum(board[:, col])) == GRID_SIZE:
            return board[0, col]
    # 检查对角线
    if abs(np.sum(np.diagonal(board))) == GRID_SIZE:
        return board[0, 0]
    if abs(np.sum(np.diagonal(np.fliplr(board)))) == GRID_SIZE:
        return board[0, 2]
    return 0  # No winner

# 重置棋盘
def reset_board():
    return np.zeros((GRID_SIZE, GRID_SIZE))

# 玩家操作
def player_move(env, player):
    valid_move = False
    while not valid_move:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.MOUSEBUTTONDOWN:
                pos = pygame.mouse.get_pos()
                row, col = pos[1] // CELL_SIZE, pos[0] // CELL_SIZE
                action = row * GRID_SIZE + col
                # Ensure the cell is empty and the move is valid
                if env.board[row, col] == 0:
                    valid_move = True
                    next_state, reward, done = env.step(action, player)
                    return next_state, reward, done

# Tic-Tac-Toe 环境
class TicTacToeEnv:
    def __init__(self):
        self.board = np.zeros((3, 3))  # 3x3 board, 0 = empty, 1 = agent, -1 = opponent
        self.done = False
        self.winner = None

    def reset(self):
        self.board = np.zeros((3, 3))
        self.done = False
        self.winner = None
        return self.board.flatten()

    def step(self, action, player):
        row, col = divmod(action, 3)
        if self.board[row, col] != 0:
            return self.board.flatten(), -10, True  # Invalid move, penalty
        self.board[row, col] = player

        if self.check_winner(player):
            self.done = True
            self.winner = player
            return self.board.flatten(), 10, self.done  # Positive reward for winning
        elif np.all(self.board != 0):
            self.done = True
            return self.board.flatten(), 0, self.done  # No winner, tie
        else:
            return self.board.flatten(), 0, self.done  # Continue playing

    def check_winner(self, player):
        for row in range(3):
            if np.all(self.board[row, :] == player):
                return True
        for col in range(3):
            if np.all(self.board[:, col] == player):
                return True
        if np.all(np.diagonal(self.board) == player):
            return True
        if np.all(np.diagonal(np.fliplr(self.board)) == player):
            return True
        return False

    def render(self):
        draw_board()
        draw_marks(self.board)  # Draw both X and O marks
        pygame.display.update()

# 游戏展示
def play_game(agent, env):
    state = env.reset()
    done = False
    env.render()
    while not done:
        # 让玩家选择行动
        next_state, reward, done = player_move(env, player=-1)  # 玩家是 -1
        draw_marks(env.board)  # Ensure to draw the player's mark
        pygame.display.update()  # Ensure the screen updates after drawing player's mark

        if done:
            print("Game Over! Player Wins!" if reward > 0 else "Game Over! It's a Tie.")
            break

        # 让智能体选择行动
        action = agent.choose_action(state)
        print(f"Agent chooses action {action}")
        next_state, reward, done = env.step(action, player=1)  # 假设智能体为1号玩家
        agent.update_q_value(state, action, reward, next_state, done)
        state = next_state
        draw_marks(env.board)  # 绘制棋盘和标记
        pygame.display.update()  # Ensure the screen updates after drawing agent's mark
        if done:
            print("Game Over! Agent Wins!" if reward > 0 else "Game Over! It's a Tie.")
            break

# Q-learning 智能体
class QLearningAgent:
    def __init__(self, action_space, epsilon=0.1, alpha=0.5, gamma=0.9):
        self.action_space = action_space
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.q_table = {}

    def get_q_value(self, state, action):
        if tuple(state) not in self.q_table:
            self.q_table[tuple(state)] = np.zeros(len(self.action_space))
        return self.q_table[tuple(state)][action]

    def update_q_value(self, state, action, reward, next_state, done):
        future_q_value = 0 if done else np.max(self.q_table.get(tuple(next_state), np.zeros(len(self.action_space))))
        current_q_value = self.get_q_value(state, action)
        new_q_value = current_q_value + self.alpha * (reward + self.gamma * future_q_value - current_q_value)
        self.q_table[tuple(state)][action] = new_q_value

    def choose_action(self, state):
        if random.uniform(0, 1) < self.epsilon:
            return random.choice(self.action_space)
        else:
            return np.argmax(self.q_table.get(tuple(state), np.zeros(len(self.action_space))))

# 主函数和游戏循环
def main():
    agent = QLearningAgent(action_space=list(range(9)))
    env = TicTacToeEnv()

    while True:
        draw_board()  # 绘制棋盘
        play_game(agent, env)  # 开始游戏

if __name__ == "__main__":
    main()
