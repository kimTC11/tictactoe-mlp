# TicTacToe Data Generator

Công cụ tạo dữ liệu huấn luyện cho Neural Network chơi TicTacToe với kích thước bàn cờ và điều kiện thắng tùy chỉnh.

## Tính năng

- ✅ Tạo tất cả các trạng thái game có thể có
- ✅ Sử dụng thuật toán Minimax để tìm nước đi tốt nhất
- ✅ Hỗ trợ kích thước bàn cờ tùy chỉnh (3x3, 4x4, 5x5, ...)
- ✅ Điều kiện thắng tùy chỉnh (3 liên tiếp, 4 liên tiếp, ...)
- ✅ Alpha-Beta pruning để tối ưu hiệu suất
- ✅ Xuất dữ liệu ra file CSV

## Cách sử dụng

### 1. Chạy chương trình

```bash
cd generate_data
python tictactoe_data_generator.py
```

### 2. Nhập thông số

- **Kích thước bàn cờ**: Ví dụ nhập `3` cho bàn cờ 3x3
- **Điều kiện thắng**: Ví dụ nhập `3` để cần 3 quân liên tiếp mới thắng

### 3. Kết quả

Chương trình sẽ tạo file CSV với tên dạng: `tictactoe_3x3_win3.csv`

## Cấu trúc dữ liệu đầu ra

### Header của CSV:

- **C1, C2, ..., CN**: Các ô trên bàn cờ (N = size × size)
- **player**: Người chơi hiện tại
  - `1` = X (người chơi)
  - `0` = O (AI)
  - `-1` = Game kết thúc
- **move**: Nước đi tốt nhất (1-based index, -1 nếu game kết thúc)

### Giá trị trong các ô:

- `1` = X (người chơi)
- `0` = O (AI)
- `-1` = Ô trống

## Ví dụ

### Bàn cờ 3x3, điều kiện thắng 3:

```
C1 | C2 | C3
-----------
C4 | C5 | C6
-----------
C7 | C8 | C9
```

### Dữ liệu mẫu:

```csv
C1,C2,C3,C4,C5,C6,C7,C8,C9,player,move
-1,-1,-1,-1,-1,-1,-1,-1,-1,1,5
1,-1,-1,-1,-1,-1,-1,-1,-1,0,5
1,-1,-1,-1,0,-1,-1,-1,-1,1,9
```

## Hiệu suất

- **3x3**: ~5,000+ trạng thái
- **4x4**: ~100,000+ trạng thái  
- **5x5**: Hàng triệu trạng thái (cần thời gian lâu)

## Lưu ý

- Với bàn cờ lớn (>4x4), thời gian tạo dữ liệu có thể rất lâu
- Sử dụng Alpha-Beta pruning để tối ưu hiệu suất
- Dữ liệu được tạo theo thuật toán Minimax (optimal play)

## Tùy chỉnh

Bạn có thể chỉnh sửa file `tictactoe_data_generator.py` để:

- Thay đổi cách tính điểm
- Thêm các chiến lược khác ngoài Minimax
- Tùy chỉnh format đầu ra
- Thêm validation cho input

## Cài đặt dependencies

```bash
pip install numpy pandas
```

Hoặc với uv:

```bash
uv add numpy pandas
```