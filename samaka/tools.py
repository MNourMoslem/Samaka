import nchess as nc
import torch

def board_to_tensor(board : nc.Board, device = None):
    return torch.from_numpy(board.as_array(), dtype = torch.float32, device = device)
