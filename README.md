# Tic-Tac-Toe with Q-learning

## 项目简介
本项目实现了一个基于 Q-learning 的井字棋（Tic-Tac-Toe）AI，能够通过强化学习不断优化自身策略，达到最佳游戏水平。

## 玩法规则
- 游戏在 3x3 棋盘上进行，玩家 X 先手，双方交替落子。
- 若有三颗棋子连成一线（行、列或对角线），该玩家获胜。
- 棋盘填满且无人获胜，则为平局。

## Q-learning 介绍
- **学习率（α）**: 控制 Q 值更新的快慢。
- **折扣因子（γ）**: 衡量未来奖励的重要性。
- **探索率（ε）**: 让 AI 在探索（随机行动）和开发（利用最佳策略）之间保持平衡。

## 安装与运行

### 1️⃣ 克隆仓库
```bash
git clone https://github.com/yzc03/Q-learning.git
cd Q-learning
## 演示视频
[点击这里观看演示视频]([https://www.youtube.com/watch?v=视频ID](https://www.youtube.com/watch?v=Ahoxl7mn6j0))

