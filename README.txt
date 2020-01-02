
書籍「ゼロから作るDeep Learning」第5章を参考に、勉強のためにc++でneural networkを実装したもの
手書き文字(mnist)の認識プログラム

アルゴリズムの概要:
・インプット - >第一隠れ層(Affineレイヤ,50ノード) -> ReLUレイヤ -> 第二隠れ層(Affineレイヤ,10ノード) -> Softmaxレイヤ -> Cross Entropy誤差
・重み関数の初期値はただのガウス分布で、平均は0、標準偏差は0.01
・重みの更新はStochastic Gradient Decent

プログラムのコンパイル方法:
・srcディレクトリ内でmake -> mainという実行ファイルが生成される
・C++の線形代数ライブラリ「Eigen(ver.3.9以降)」を使用しているので、Eigenのダウンロードと、インクルードパスへの追加が別途必要

プログラムの実行方法:
・./main x_train.csv t_train.csv x_test.csv t_test.csv
  (トレーニングデータファイルはdataディレクトリ内にある)

プログラムの構成:
・srcディレクトリ
  プログラムが入っている。main.cppの中にmain関数があり、makefileによってmainというプログラムが生成される。
・data.zip
  手書き文字(mnist)のトレーニングデータが入っている。
  ・x_train.csv  手書き文字のトレーニングデータ
  ・t_train.csv  手書き文字の正解ラベル(One-Hot encoding済み)のトレーニングデータ
  ・x_test.csv   手書き文字のテストデータ
  ・t_test.csv   手書き文字の正解ラベルのテストデータ
