import chess
import torch.nn as nn

from .model import *
from .tools import *

def find_best_move(board : chess.Board, model : nn.Module, maximizing_player : int):
  best_move = None
  device = model.device

  if maximizing_player:
    best_val = float('-inf')
    
    for move in board.legal_moves:
      board.push(move)
      tboard = board_to_tensor(board, device)
      val = model(tboard)
      if val > best_val:
        best_val = val
        best_move = move
      board.pop()
    
  else:
    best_val = float('inf')
    
    for move in board.legal_moves:
      board.push(move)
      tboard = board_to_tensor(board, device)
      val = model(tboard)
      if val < best_val:
        best_val = val
        best_move = move
      board.pop()

  return best_move, best_val

def play_game(model : nn.Module, board : chess.Board, until_num_move : int = -1):
  if until_num_move == -1:
    until_end = True
  else:
    until_end = False
  num_move = 0

  if until_end:
    while not board.is_game_over():
      best_move, _ = find_best_move(board, model, board.turn)
      board.push(best_move)
      num_move += 1
  else:
    for _ in range(until_num_move):
      best_move, _ = find_best_move(board, model, board.turn)
      board.push(best_move)
      num_move += 1

  game_result = board.result()
  return board, num_move, game_result