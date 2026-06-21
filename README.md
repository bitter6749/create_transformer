# C-Transformer: プロジェクト構造とデータフロー解説書
C言語のみを使用し、外部ライブラリを一切使わずにゼロから実装している、文字レベルの超軽量Transformer（Decoderブロック 1層分）の解説テキストです。
ポインタによるメモリ管理、行列演算の縦横スキャン、自動微分（バックプロパゲーション）の仕組みを本質から整理することを目的としています。


# 全体構造とデータフロー (Architecture)

---

本プロジェクトでは、入力された文字（例: "cat " の4文字）から、「次に来る5文字目の確率分布」を予測する、Transformerの核心部「Self-Attention層」を実装しています。

## 順伝播 (Forward Propagation)
入力データは以下のステップを経て、数式通りに行列へと変換されていきます。

## JEmbedding層 (X の生成)

離散的な文字ID（VOCAB_SIZE=27）を、連続的な意味を持つベクトル（EMBED_DIM=16）に変換します。

```
input_ids [4] -> X [4 x 16]
```

## Linear層 (Q, K, V の生成)

入力 X に対して、それぞれの役割（Query, Key, Value）を持つ重み行列を掛け合わせます。

```
Q = X ・ W_q

K = X ・ W_k

V = X ・ W_v
```

## Self-Attention層 (Z の生成)

相性表の計算 (A):
クエリ Q と キー K の内積を取り、次元数のルート（√16 = 4）でスケール調整をした後、横方向に Softmax をかけて確率化します。
```
数式: A = Softmax( (Q ・ K^T) / √d_k )
```

価値 of ブレンド (Z):
計算した相性表（注目度）に従って、Valueのベクトルを混ぜ合わせます。
``` 
数式: Z = A ・ V
```

## Output層 (次の一文字の予測確率)

文脈情報を最も吸い込んでいる「最後の文字（4文字目のスペース）」のベクトル last_z だけを抽出し、出力用の重み W_out を掛けてSoftmaxを通し、27文字の確率分布を吐き出します。

```
output = Softmax( last_z ・ W_out^T )
```

---

## 逆伝播 (Backward Propagation)
学習時は、上記のフローを完全に逆順に辿りながら、上流から流れてくる「予測と正解の誤差（勾配）」を下流の関数へとバトンタッチしていきます。

```
[Outputの誤差]
│
▼
backward_output()       # W_out の勾配を計算、dZ を上流へ
│
▼
backward_attention_qkv() # dA の計算、Softmaxの微分、dQ と dK を上流へ
│
▼
(Next Step...)          # W_q, W_k, W_v への勾配計算と、重みの更新
```

---

## ディレクトリ構造 (Directory Map)
オブジェクト指向的に、データの構造（Matrix）と、ニューラルネットワークの層（Transformer）、独立した検証環境（Test）がそれぞれ拡張しやすいよう設計されています。

```
├── Makefile                # Windows(devkit)/Linux 両対応の自動化ビルド環境
├── include/
│   ├── matrix.h            # 行列の構造体定義と基本演算（転置積など）
│   ├── model.h             # SimpleTransformer モデルの構造体定義
│   └── transformer.h       # 各層の Forward / Backward 関数の宣言
├── src/
│   ├── main.c              # アプリのメインエントリー（通常の予測実行）
│   ├── matrix.c            # 行列演算の実装（ポインタスキャン）
│   ├── model.c             # モデルの初期化とメモリ解放
│   └── transformer.c       # 各層の具体的な計算ロジック
└── test/                   # 拡張性抜群の独立ユニットテスト環境
├── test_main.c         # テストの統括・実行（エントリーポイント）
├── test_matrix.c       # 行列演算が正しくスキャンできているかの検証
├── test_softmax.c      # Softmax単体の数値検証
├── test_backward_attentin.c      # Attention (q,k,vの逆伝播)
└── test_attention.c    # Attention（順伝播・逆伝播）
```

--- 

## コマンド集 (How to Run)
自作した Makefile の自動化タスクにより、手動でファイルを指定してコンパイルする必要はありません。

## 通常版のビルドと実行
モデルを起動し、前方伝播を行って次の文字を予測します。
```sh
make run
./bin/transformer
```

## ユニットテストの実行
テストコードを自動検知してビルドし、PyTorchの計算結果と1マス残らず数値が一致するかを厳密に検証します。
```sh
make test
```


## プロジェクトのクリーンアップ
ビルドによって生成された一時ファイル（.o）や実行ファイルを削除し、まっさらな状態に戻します。
```sh
make clean
```