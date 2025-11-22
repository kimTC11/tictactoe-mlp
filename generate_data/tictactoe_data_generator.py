import itertools
import csv

# --- Symmetry operations for canonical reduction ---
def rotate(board, n):
    return [board[(n - 1 - c) * n + r] for r in range(n) for c in range(n)]

def reflect(board, n):
    return [board[r * n + (n - 1 - c)] for r in range(n) for c in range(n)]


def all_symmetries(board, n):
    b = board
    rots = [b]
    for _ in range(3):
        b = rotate(b, n)
        rots.append(b)
    result = []
    for r in rots:
        result.append(r)
        result.append(reflect(r, n))
    return result


def canonical_form(board, n):
    syms = all_symmetries(board, n)
    return tuple(min(syms))

# --- Game logic ---
def check_win(board, n, player):
    # rows
    for r in range(n):
        if all(board[r*n + c] == player for c in range(n)):
            return True
    # cols
    for c in range(n):
        if all(board[r*n + c] == player for r in range(n)):
            return True
    # diag
    if all(board[i*n + i] == player for i in range(n)):
        return True
    # anti-diag
    if all(board[i*n + (n-1-i)] == player for i in range(n)):
        return True
    return False


def next_player(board):
    x = board.count(1)
    o = board.count(0)
    if x == o:
        return 1
    if x == o + 1:
        return 0
    return -1

# Determine move field
# -1 = no move
# 'D' = draw
# terminal win → move = -1

def detect_move(board, n):
    # Win conditions
    if check_win(board, n, 1):
        return -1, -1
    if check_win(board, n, 0):
        return -1, -1
    # Draw
    if -1 not in board:
        return -1, 'D'
    # Ongoing
    return next_player(board), -1

# --- Generate canonical states ---
def is_valid_state(board):
    x = board.count(1)
    o = board.count(0)
    # invalid stone count
    if not (x == o or x == o + 1):
        return False
    # both cannot win
    if check_win(board, n, 1) and check_win(board, n, 0):
        return False
    # if X wins → must have x==o+1
    if check_win(board, n, 1) and x != o + 1:
        return False
    # if O wins → must have x==o
    if check_win(board, n, 0) and x != o:
        return False
    return True


def generate_states(n):
    size = n * n
    seen = set()
    canonical_states = []

    for board in itertools.product([-1, 0, 1], repeat=size):
        board = list(board)
        if not is_valid_state(board):
            continue

        cf = canonical_form(board, n)
        if cf in seen:
            continue
        seen.add(cf)

        player, move = detect_move(board, n)

        canonical_states.append((list(cf), player, move))

    return canonical_states


# --- Write CSV ---
def write_csv(states, path, n):
    headers = [f"c{i+1}" for i in range(n*n)] + ["player", "move"]

    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for state, player, move in states:
            writer.writerow(state + [player, move])


if __name__ == "__main__":
    import sys
    n = int(sys.argv[1])
    out = sys.argv[2]
    states = generate_states(n)
    write_csv(states, out, n)
    print(f"Generated {len(states)} canonical states → {out}")
