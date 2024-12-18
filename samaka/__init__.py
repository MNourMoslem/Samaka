import chess
import torch.nn as nn

from .model import *
from .tools import *

def find_best_move(board : chess.Board, model : nn.Module, maximizing_player : int):
  best_move = None
  
  if maximizing_player:
    best_val = float('-inf')
    
    for move in board.legal_moves:
      board.push(move)
      tboard = board_to_tensor(board)
      val = model(tboard)
      if val > best_val:
        best_val = val
        best_move = move
      board.pop()
    
  else:
    best_val = float('inf')
    
    for move in board.legal_moves:
      board.push(move)
      tboard = board_to_tensor(board)
      val = model(tboard)
      if val < best_val:
        best_val = val
        best_move = move
      board.pop()

  return best_move, best_val