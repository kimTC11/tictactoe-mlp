import pandas as pd

# Load file
df = pd.read_csv("tictactoe_games.csv")

# Board mapping function
def pos_to_index(r, c):
    return r * 3 + c

rows_out = []
for _, row in df.iterrows():
    board = [-1] * 9
    moves = [col for col in df.columns if "Move" in col]
    
    player_turn = 1  # X starts
    
    for m in moves:
        mv = row[m]
        if mv == "---":
            break
        r, c = map(int, mv.split("-"))
        idx = pos_to_index(r, c)
        board[idx] = player_turn
        player_turn = 0 if player_turn == 1 else 1  # toggle
        
    # terminal state
    player = -1
    move = -1
    
    rows_out.append(board + [player, move])

out_df = pd.DataFrame(rows_out, columns=[f"c{i+1}" for i in range(9)] + ["player", "move"])
out_path = "converted_canonical.csv"
out_df.to_csv(out_path, index=False)

out_path
