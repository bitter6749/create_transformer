# 使用するコンパイラをすべて nvcc (Microsoft cl.exe 駆動) に統合
CC = nvcc
NVCC = nvcc

# インクルードパスと最適化フラグ
# -Xcompiler は裏側の cl.exe に直接フラグを渡すための nvcc 特有のオプションです
CFLAGS = -Iinclude -O3 -arch=sm_86 -Xcompiler "/wd4819 /wd5297"

# ディレクトリの設定
SRC_DIR = src
BUILD_DIR = build
APPS_DIR = app
BIN_DIR = bin
CUR_DIR = C:\Users\mazet\dev\transformer

# Visual Studio の環境バッチのパス (一連のコマンドをこれ経由で叩く)
VS_ENV = cd /d "C:/Program Files/Microsoft Visual Studio/18/Community/VC/Auxiliary/Build" && vcvars64.bat && cd $(CUR_DIR)

# 最終的な実行ファイルの名前と場所
TARGET_TRAIN = $(BIN_DIR)/train
TARGET_GEN = $(BIN_DIR)/generate

# src/ フォルダにあるすべての .c ファイルと .cu ファイル
SRCS_C = $(wildcard $(SRC_DIR)/*.c)
OBJS_C = $(patsubst $(SRC_DIR)/%.c, $(BUILD_DIR)/%.o, $(SRCS_C))

# 最初にする命令
all: make_dirs $(TARGET_TRAIN) $(TARGET_GEN)

# 必要なフォルダを作成
make_dirs:
	@mkdir -p $(BUILD_DIR) $(BIN_DIR)

# 1. GPU用コードのコンパイル
$(BUILD_DIR)/ops_cuda.o: src/ops_cuda.cu | make_dirs
	@cmd /c "$(VS_ENV) && $(NVCC) $(CFLAGS) -c $< -o $@"

# 2. すべての .c ファイルを nvcc でコンパイル (MinGWの方言を排除)
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.c | make_dirs
	@cmd /c "$(VS_ENV) && $(CC) $(CFLAGS) -c $< -o $@"

# 3. 最終リンク（すべて同じ Microsoft 規格なので 100% 綺麗に合体します）
$(TARGET_TRAIN): $(OBJS_C) $(BUILD_DIR)/ops_cuda.o $(APPS_DIR)/train.c
	@cmd /c "$(VS_ENV) && $(NVCC) $(CFLAGS) $(OBJS_C) $(BUILD_DIR)/ops_cuda.o $(APPS_DIR)/train.c -o $@"
	@echo ">>> リンク完了: $(TARGET_TRAIN)"

train: $(TARGET_TRAIN)
	./$(TARGET_TRAIN)

$(TARGET_GEN): $(OBJS_C) $(BUILD_DIR)/ops_cuda.o $(APPS_DIR)/generate.c
	@cmd /c "$(VS_ENV) && $(NVCC) $(CFLAGS) $(OBJS_C) $(BUILD_DIR)/ops_cuda.o $(APPS_DIR)/generate.c -o $@"
	@echo ">>> テキストジェネレータ リンク完了: $(TARGET_GEN)"

generate: $(TARGET_GEN)
	./$(TARGET_GEN)

# 生成されたビルド物や実行ファイルを削除するコマンド
clean:
	rm -rf build bin
	@echo "--- Cleanup completed ---"

re: clean all