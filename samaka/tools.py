import chess
import torch

if chess.WHITE == 1 and chess.BLACK == 0:
    get_color = lambda x: x
else:
    color_to_num = {
        chess.WHITE: 1,
        chess.BLACK: 0,
    }
    get_color = lambda x: color_to_num[x]

piece_to_number = {
    chess.PAWN: 1,
    chess.KNIGHT: 2,
    chess.BISHOP: 3,
    chess.ROOK: 4,
    chess.QUEEN: 5,
    chess.KING: 6,
}

def board_to_tensor(board):
    tensor = torch.zeros(64, dtype=torch.int64)

    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            base_number = piece_to_number[piece.piece_type]
            color_offset = 6 * get_color(piece.color)
            tensor[square] = base_number + color_offset

    return tensor
