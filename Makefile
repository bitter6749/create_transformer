# 使用するコンパイラ
CC = gcc
# -Wall -Wextra 警告をすべて表示
# -pedantic 公式標準規格に則っているコードかチェック
CFALGS = -Wall -Wextra -pedantic -Iinclude

# 最終的な実行ファイルの名前と場所
TARGET = bin/transformer

# src/ フォルダにあるすべての .c ファイルを自動でリストアップ
CRCS = $(wildcard src/*.c)

# リストアップした .c ファイルの名前を build/ 内の .oファイル名に自動変換する
OBJS = $(patsubst src/%.c, build/%.o, $(SRCS))


# 最初にする命令
# フォルダを作って、すべてコンパイルして、リンクする
all: make_dirs $(OBJS) link_all

# 必要なフォルダを作成
make_dirs:
	mkdir -p build bin

# src/ 内のすべての .c から、build/ 内の .oを作る
# % はワイルドカード (任意の文字列) を意味する
# $< はコンパイル対象の.cファイル
# $@ は出力先の.oファイルを指す
build/%.o: src/%.c
	$(CC) $(CFALGS) -c $< -o $@

# 依存関係に $(OBJS)を指定したため
# .oファイルのどれか一つでもタイムスタンプが新しくなった時だけ、リンク処理が走る
$(TARGET): $(OBJS)
	$(CC) (OBJS) -o $(TARGET) -lm
	@echo ">>> 変更を検知したため、再リンクを実行しました: $(TARGET)"

# 生成されたビルド物や実行ファイルを削除するコマンド
clean:
	rm -rf build bin
	@echo "--- クリーンアップが完了しました ---"

# クリーンな状態にしてビルドする
re: clean all
	
# 実行する
run: $(TARGET) 
	./$(TARGET)

