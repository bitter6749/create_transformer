# 使用するコンパイラ
CC = gcc
# -Wall -Wextra 警告をすべて表示
# -pedantic 公式標準規格に則っているコードかチェック
CFALGS = -Wall -Wextra -pedantic -Iinclude

# ディレクトリの設定
SRC_DIR = src
BUILD_DIR = build

# 最終的な実行ファイルの名前と場所
TARGET = bin/transformer

# src/ フォルダにあるすべての .cファイルを自動でリストアップ
SRCS = $(wildcard $(SRC_DIR)/*.c)

# リストアップした .c ファイルの名前を build/ 内の .oファイル名に自動変換する
OBJS = $(patsubst $(SRC_DIR)/%.c, $(BUILD_DIR)/%.o, $(SRCS))

# 最初にする命令
# フォルダを作って、すべてコンパイルして、リンクする
# $(TARGET) を指定して、実行ファイルを作るための処理を Makefile に実行させる
all: make_dirs $(TARGET)

# 必要なフォルダを作成
make_dirs:
	mkdir -p build bin

# src/ 内のすべての .c から、build/ 内の .oを作る
# % はワイルドカード (任意の文字列) を意味する
# $< はコンパイル対象の.cファイル
# $@ は出力先の.oファイルを指す
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.c | make_dirs
	$(CC) $(CFALGS) -c $< -o $@

# 依存関係に $(OBJS)を指定したため
# .oファイルのどれか一つでもタイムスタンプが新しくなった時だけ、リンク処理が走る
$(TARGET): $(OBJS)
	$(CC) $(OBJS) -o $(TARGET) -lm
	@echo ">>> A change was detected and the link was re-linked: $(TARGET)"

# 生成されたビルド物や実行ファイルを削除するコマンド
clean:
	rm -rf build bin
	@echo "--- Cleanup completed ---"

# クリーンな状態にしてビルドする
re: clean all
	
# 実行する
run: $(TARGET) 
	./$(TARGET)

