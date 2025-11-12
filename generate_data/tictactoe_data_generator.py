"""
TicTacToe Data Generator
Tạo tất cả các trạng thái và nước đi có thể có cho trò chơi TicTacToe
với kích thước bàn cờ và điều kiện thắng tùy chỉnh.

Input:
- size: Kích thước bàn cờ (ví dụ: 3 cho 3x3)
- win_condition: Số quân liên tiếp cần để thắng (ví dụ: 3)

Output:
- CSV file chứa tất cả các trạng thái và nước đi tốt nhất
"""

import numpy as np
import pandas as pd
from itertools import combinations_with_replacement
import csv
from typing import List, Tuple, Optional
import os

class TicTacToeDataGenerator:
    def __init__(self, size: int = 3, win_condition: int = 3):
        """
        Khởi tạo generator
        
        Args:
            size: Kích thước bàn cờ (size x size)
            win_condition: Số quân liên tiếp cần để thắng
        """
        self.size = size
        self.win_condition = win_condition
        self.total_cells = size * size
        
        # Định nghĩa giá trị
        self.EMPTY = -1
        self.PLAYER_O = 0  # AI
        self.PLAYER_X = 1  # Human
        
        self.game_states = []
        
    def get_all_lines(self) -> List[List[int]]:
        """
        Lấy tất cả các đường thẳng có thể thắng trên bàn cờ
        """
        lines = []
        
        # Hàng ngang
        for row in range(self.size):
            for start_col in range(self.size - self.win_condition + 1):
                line = [row * self.size + start_col + i for i in range(self.win_condition)]
                lines.append(line)
        
        # Cột dọc
        for col in range(self.size):
            for start_row in range(self.size - self.win_condition + 1):
                line = [(start_row + i) * self.size + col for i in range(self.win_condition)]
                lines.append(line)
        
        # Đường chéo chính (từ trái trên xuống phải dưới)
        for row in range(self.size - self.win_condition + 1):
            for col in range(self.size - self.win_condition + 1):
                line = [(row + i) * self.size + (col + i) for i in range(self.win_condition)]
                lines.append(line)
        
        # Đường chéo phụ (từ phải trên xuống trái dưới)
        for row in range(self.size - self.win_condition + 1):
            for col in range(self.win_condition - 1, self.size):
                line = [(row + i) * self.size + (col - i) for i in range(self.win_condition)]
                lines.append(line)
        
        return lines
    
    def check_winner(self, board: List[int]) -> Optional[int]:
        """
        Kiểm tra người thắng
        
        Returns:
            PLAYER_X nếu X thắng
            PLAYER_O nếu O thắng  
            None nếu chưa có người thắng
        """
        lines = self.get_all_lines()
        
        for line in lines:
            if all(board[pos] == self.PLAYER_X for pos in line):
                return self.PLAYER_X
            if all(board[pos] == self.PLAYER_O for pos in line):
                return self.PLAYER_O
        
        return None
    
    def is_board_full(self, board: List[int]) -> bool:
        """Kiểm tra bàn cờ đã đầy chưa"""
        return self.EMPTY not in board
    
    def get_empty_positions(self, board: List[int]) -> List[int]:
        """Lấy danh sách vị trí trống"""
        return [i for i in range(self.total_cells) if board[i] == self.EMPTY]
    
    def minimax(self, board: List[int], is_maximizing: bool, alpha: float = float('-inf'), beta: float = float('inf')) -> int:
        """
        Thuật toán Minimax với Alpha-Beta pruning để tìm nước đi tốt nhất
        
        Returns:
            Điểm số của trạng thái hiện tại
        """
        winner = self.check_winner(board)
        
        # Trường hợp cơ sở
        if winner == self.PLAYER_O:  # AI thắng
            return 10
        elif winner == self.PLAYER_X:  # Human thắng
            return -10
        elif self.is_board_full(board):  # Hòa
            return 0
        
        empty_positions = self.get_empty_positions(board)
        
        if is_maximizing:  # Lượt AI (O)
            max_eval = float('-inf')
            for pos in empty_positions:
                board[pos] = self.PLAYER_O
                eval_score = self.minimax(board, False, alpha, beta)
                board[pos] = self.EMPTY
                max_eval = max(max_eval, eval_score)
                alpha = max(alpha, eval_score)
                if beta <= alpha:
                    break  # Alpha-beta pruning
            return max_eval
        else:  # Lượt Human (X)
            min_eval = float('inf')
            for pos in empty_positions:
                board[pos] = self.PLAYER_X
                eval_score = self.minimax(board, True, alpha, beta)
                board[pos] = self.EMPTY
                min_eval = min(min_eval, eval_score)
                beta = min(beta, eval_score)
                if beta <= alpha:
                    break  # Alpha-beta pruning
            return min_eval
    
    def get_best_move(self, board: List[int], player: int) -> int:
        """
        Tìm nước đi tốt nhất cho người chơi hiện tại
        
        Returns:
            Vị trí tốt nhất để đi (0-based index), hoặc -1 nếu game đã kết thúc
        """
        # Kiểm tra xem game đã kết thúc chưa
        if self.check_winner(board) is not None or self.is_board_full(board):
            return -1
        
        empty_positions = self.get_empty_positions(board)
        best_move = -1
        
        if player == self.PLAYER_O:  # AI turn
            best_score = float('-inf')
            for pos in empty_positions:
                board[pos] = self.PLAYER_O
                score = self.minimax(board, False)
                board[pos] = self.EMPTY
                if score > best_score:
                    best_score = score
                    best_move = pos
        else:  # Human turn (X)
            best_score = float('inf')
            for pos in empty_positions:
                board[pos] = self.PLAYER_X
                score = self.minimax(board, True)
                board[pos] = self.EMPTY
                if score < best_score:
                    best_score = score
                    best_move = pos
        
        return best_move
    
    def generate_all_states(self):
        """Tạo tất cả các trạng thái có thể có của game"""
        self.game_states = []
        
        # Bắt đầu với bàn cờ trống
        initial_board = [self.EMPTY] * self.total_cells
        self._generate_states_recursive(initial_board, self.PLAYER_X)  # X đi trước
        
        print(f"Đã tạo {len(self.game_states)} trạng thái game")
    
    def _generate_states_recursive(self, board: List[int], current_player: int):
        """Đệ quy để tạo tất cả các trạng thái"""
        # Kiểm tra xem game đã kết thúc chưa
        winner = self.check_winner(board)
        if winner is not None or self.is_board_full(board):
            # Game kết thúc, lưu trạng thái
            best_move = -1  # Không có nước đi
            next_player = -1  # Game kết thúc
            self.game_states.append({
                'board': board.copy(),
                'player': next_player,
                'best_move': best_move
            })
            return
        
        # Tìm nước đi tốt nhất cho trạng thái hiện tại
        best_move = self.get_best_move(board.copy(), current_player)
        
        # Lưu trạng thái hiện tại
        self.game_states.append({
            'board': board.copy(),
            'player': current_player,
            'best_move': best_move
        })
        
        # Thử tất cả các nước đi có thể
        empty_positions = self.get_empty_positions(board)
        for pos in empty_positions:
            new_board = board.copy()
            new_board[pos] = current_player
            next_player = self.PLAYER_O if current_player == self.PLAYER_X else self.PLAYER_X
            self._generate_states_recursive(new_board, next_player)
    
    def save_to_csv(self, filename: str):
        """Lưu dữ liệu vào file CSV"""
        if not self.game_states:
            print("Chưa có dữ liệu. Hãy chạy generate_all_states() trước.")
            return
        
        # Tạo header cho các ô
        headers = [f'C{i+1}' for i in range(self.total_cells)]
        headers.extend(['player', 'move'])
        
        # Tạo file CSV
        with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(headers)
            
            for state in self.game_states:
                row = state['board'].copy()
                row.extend([state['player'], state['best_move']])
                writer.writerow(row)
        
        print(f"Đã lưu {len(self.game_states)} trạng thái vào file: {filename}")
    
    def print_board(self, board: List[int]):
        """In bàn cờ ra console để debug"""
        symbols = {self.EMPTY: '_', self.PLAYER_O: 'O', self.PLAYER_X: 'X'}
        
        for row in range(self.size):
            row_str = []
            for col in range(self.size):
                pos = row * self.size + col
                row_str.append(symbols[board[pos]])
            print(' | '.join(row_str))
            if row < self.size - 1:
                print('-' * (4 * self.size - 1))
        print()


def main():
    """Hàm chính để chạy generator"""
    print("=== TicTacToe Data Generator ===")
    
    # Nhập thông số từ người dùng
    try:
        size = int(input("Nhập kích thước bàn cờ (mặc định 3): ") or 3)
        win_condition = int(input(f"Nhập số quân liên tiếp để thắng (mặc định {size}): ") or size)
        
        if win_condition > size:
            win_condition = size
            print(f"Điều kiện thắng không thể lớn hơn kích thước bàn cờ. Đã điều chỉnh thành {size}")
        
    except ValueError:
        print("Sử dụng giá trị mặc định: 3x3, điều kiện thắng 3")
        size = 3
        win_condition = 3
    
    # Tạo generator
    generator = TicTacToeDataGenerator(size, win_condition)
    
    # Tạo dữ liệu
    print(f"\nBắt đầu tạo dữ liệu cho bàn cờ {size}x{size} với điều kiện thắng {win_condition}...")
    generator.generate_all_states()
    
    # Lưu vào file
    filename = f"tictactoe_{size}x{size}_win{win_condition}.csv"
    filepath = os.path.join(os.path.dirname(__file__), filename)
    generator.save_to_csv(filepath)
    
    print(f"\nHoàn thành! File dữ liệu: {filepath}")
    print(f"Tổng số trạng thái: {len(generator.game_states)}")


if __name__ == "__main__":
    main()