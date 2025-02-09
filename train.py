import torch
from torch import nn
import nchess as nc
from collections import deque
import copy
from samaka.tools import board_to_tensor
import random

def find_move_epsilon(board : nc.Board, model : nn.Module, maximizing_player : int, epsilon : float, device : torch.device = None):
    best_move = None
    random_move = random.random()

    legal_moves = board.generate_legal_moves()

    if epsilon > 0 and random_move < epsilon:
        best_move = random.choice(legal_moves)
        board.step(best_move)
        tboard = board_to_tensor(board, device)
        best_val = model(tboard)    
        board.undo()
        return best_move, best_val

    if maximizing_player:
        best_val = float('-inf')
        
        for move in legal_moves:
            board.step(move)
            tboard = board_to_tensor(board, device)
            val = model(tboard)
            if val > best_val:
                best_val = val
                best_move = move
            board.undo()
        
    else:
        best_val = float('inf')
        
        for move in legal_moves:
            board.step(move)
            tboard = board_to_tensor(board, device)
            val = model(tboard)
            if val < best_val:
                best_val = val
                best_move = move
            board.undo()

    return best_move, best_val

def train(model : nn.Module, optimizer, loss_func, epsilon, device : torch.device = None, epochs : int = 1000, gamma : float = 0.99):

    trg_model = copy.deepcopy(model)

    states = deque(maxlen=100000)
    next_states = deque(maxlen=100000)
    actions = deque(maxlen=100000)
    pred_values = deque(maxlen=100000)

    for epoch in range(epochs):
        board = chess.Board()

        num_moves = 0
        while not board.is_game_over():
            best_move, best_val = find_move_epsilon(board, model, board.turn, epsilon, device)

            states.append(board_to_tensor(board, device))
            actions.append(best_move)
            pred_values.append(best_val)

            board.push(best_move)

            next_states.append(board_to_tensor(board, device))
            num_moves += 1

        result = board.result()

        if result == '1-0':
            reward = 10.0
        elif result == '0-1':
            reward = -10.0
        else:
            reward = 0.0


        for i in range(num_moves):
            s, ns = states.pop(), next_states.pop()
            a, pv = actions.pop(), pred_values.pop()