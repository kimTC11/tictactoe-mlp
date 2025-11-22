"""  
TicTacToe Data Generator - User First Version  
Generates training data where user (X) plays first  
Based on minimax algorithm from Minimax_TTT.py  
"""  
  
import numpy as np  
import pandas as pd  
from itertools import product  
  
  
class TicTacToeDataGenerator:  
    """Generate optimal TicTacToe training data using minimax algorithm"""  
      
    def __init__(self):  
        self.PLAYER = 'X'    # Human player (1)  
        self.COMPUTER = 'O'  # AI player (0)  
        self.EMPTY = ' '     # Empty cell (-1)  
          
    def is_winner(self, board, player):  
        """Check if a player has won"""  
        win_patterns = [  
            [0, 1, 2], [3, 4, 5], [6, 7, 8],  # Rows  
            [0, 3, 6], [1, 4, 7], [2, 5, 8],  # Columns  
            [0, 4, 8], [2, 4, 6]              # Diagonals  
        ]  
        return any(all(board[i] == player for i in pattern) for pattern in win_patterns)  
      
    def is_moves_left(self, board):  
        """Check if there are empty cells"""  
        return self.EMPTY in board  
      
    def minimax(self, board, is_maximizing):  
        """Minimax algorithm to evaluate board positions"""  
        # Terminal states  
        if self.is_winner(board, self.COMPUTER):  
            return 10  
        if self.is_winner(board, self.PLAYER):  
            return -10  
        if not self.is_moves_left(board):  
            return 0  
          
        if is_maximizing:  
            best_score = float('-inf')  
            for i in range(9):  
                if board[i] == self.EMPTY:  
                    board[i] = self.COMPUTER  
                    score = self.minimax(board, False)  
                    board[i] = self.EMPTY  
                    best_score = max(score, best_score)  
            return best_score  
        else:  
            best_score = float('inf')  
            for i in range(9):  
                if board[i] == self.EMPTY:  
                    board[i] = self.PLAYER  
                    score = self.minimax(board, True)  
                    board[i] = self.EMPTY  
                    best_score = min(score, best_score)  
            return best_score  
      
    def get_best_move_for_player(self, board, player):  
        """Find the best move for specified player (X or O)"""  
        if player == self.COMPUTER:  
            # Computer maximizes  
            best_score = float('-inf')  
            best_move = -1  
              
            for i in range(9):  
                if board[i] == self.EMPTY:  
                    board[i] = self.COMPUTER  
                    score = self.minimax(board, False)  
                    board[i] = self.EMPTY  
                    if score > best_score:  
                        best_score = score  
                        best_move = i  
        else:  
            # Player minimizes (from computer's perspective)  
            best_score = float('inf')  
            best_move = -1  
              
            for i in range(9):  
                if board[i] == self.EMPTY:  
                    board[i] = self.PLAYER  
                    score = self.minimax(board, True)  
                    board[i] = self.EMPTY  
                    if score < best_score:  
                        best_score = score  
                        best_move = i  
          
        return best_move  
      
    def board_to_numeric(self, board):  
        """Convert board representation to numeric format (-1, 0, 1)"""  
        mapping = {self.EMPTY: -1, self.COMPUTER: 0, self.PLAYER: 1}  
        return [mapping[cell] for cell in board]  
      
    def is_valid_state(self, board):  
        """Check if board state is valid (proper turn order, no double wins)"""  
        x_count = board.count(self.PLAYER)  
        o_count = board.count(self.COMPUTER)  
          
        # X goes first, so x_count should be equal or one more than o_count  
        if x_count < o_count or x_count > o_count + 1:  
            return False  
          
        # Check if both players have won (invalid)  
        if self.is_winner(board, self.PLAYER) and self.is_winner(board, self.COMPUTER):  
            return False  
          
        # If someone has won, game should have stopped  
        if self.is_winner(board, self.PLAYER):  
            return x_count == o_count or x_count == o_count + 1  
        if self.is_winner(board, self.COMPUTER):  
            return x_count == o_count + 1  
          
        return True  
      
    def generate_all_states(self):  
        """Generate all valid game states for both players"""  
        data = []  
        total_configs = 3 ** 9  
        processed = 0  
          
        print(f"Processing {total_configs:,} possible board configurations...")  
          
        for config in product([self.EMPTY, self.PLAYER, self.COMPUTER], repeat=9):  
            board = list(config)  
            processed += 1  
              
            if processed % 10000 == 0:  
                print(f"  Processed {processed:,}/{total_configs:,} configurations...")  
              
            if not self.is_valid_state(board):  
                continue  
              
            # Skip if game is already over  
            if self.is_winner(board, self.PLAYER) or self.is_winner(board, self.COMPUTER):  
                numeric_board = self.board_to_numeric(board)  
                data.append(numeric_board + [-1, -1])  # No valid move  
                continue  
              
            # Skip if board is full (draw)  
            if not self.is_moves_left(board):  
                numeric_board = self.board_to_numeric(board)  
                data.append(numeric_board + [-1, -1])  
                continue  
              
            # Determine whose turn it is based on piece count  
            x_count = board.count(self.PLAYER)  
            o_count = board.count(self.COMPUTER)  
              
            numeric_board = self.board_to_numeric(board)  
              
            if x_count == o_count:  
                # X's turn (user plays first)  
                best_move = self.get_best_move_for_player(board, self.PLAYER)  
                data.append(numeric_board + [1, best_move + 1])  # player=1 for X  
            else:  
                # O's turn (computer)  
                best_move = self.get_best_move_for_player(board, self.COMPUTER)  
                data.append(numeric_board + [0, best_move + 1])  # player=0 for O  
          
        print(f"✓ Completed processing all configurations")  
        return data  
      
    def save_to_csv(self, filename='user_first_tictactoe_data.csv'):  
        """Generate and save data to CSV file"""  
        print("=" * 60)  
        print("TicTacToe Data Generator - User First Version")  
        print("=" * 60)  
          
        data = self.generate_all_states()  
          
        columns = ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'player', 'move']  
        df = pd.DataFrame(data, columns=columns)  
          
        # Save to CSV  
        df.to_csv(filename, index=False)  
          
        # Print statistics  
        print("\n" + "=" * 60)  
        print("Generation Complete!")  
        print("=" * 60)  
        print(f"Total game states generated: {len(df):,}")  
        print(f"Player X (user) states: {len(df[df['player'] == 1]):,}")  
        print(f"Player O (computer) states: {len(df[df['player'] == 0]):,}")  
        print(f"Terminal states (no move): {len(df[df['move'] == -1]):,}")  
        print(f"Output file: {filename}")  
        print("=" * 60)  
          
        return df  
  
  
def main():  
    """Main function to run the data generator"""  
    # Create generator instance  
    generator = TicTacToeDataGenerator()  
      
    # Generate and save data  
    df = generator.save_to_csv('user_first_tictactoe_data.csv')  
      
    # Display sample data  
    print("\nFirst 10 rows of generated data:")  
    print(df.head(10))  
      
    print("\nLast 10 rows of generated data:")  
    print(df.tail(10))  
      
    print("\nData format explanation:")  
    print("- C1-C9: Board state (-1=empty, 0=O, 1=X)")  
    print("- player: Current player (0=O/computer, 1=X/user)")  
    print("- move: Optimal move position (1-9, or -1 if game over)")  
  
  
if __name__ == "__main__":  
    main()