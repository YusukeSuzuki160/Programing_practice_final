# プログラミング基礎演習　最終課題

## J4200359 鈴木佑典
## 1.導入

本レポートは、東京大学教養学部2Aセメスターeeic内定コースの授業「プログラミング基礎演習」の期末レポートである。本課題では、指定の通りICAを実装(ICA.py)、それを用いて各課題を実装(kadai0X.ipynb)するという構成になっている。利用したライブラリは、"numpy", "scipy", "matplotlib", "IPython", "PIL"である。課題1～3まで実装。

## 2.手法・結果

時間もしくは位置の関数である、与えられたデータに対して、それぞれの時刻および位置における観測データを一つのまとまりとするデータのarrayを作成。それを、Report2021.pdfにある手法に従って独立成分分析する。この際、繰り返しアルゴリズム中のwが符号のみ異なっていて振動しているものについては収束と判定した。これについては考察の項で考察を行う。以下は、各課題の結果である。
- 課題1
 
  方形波と正弦波が解として得られた。実際、テスト用のデータからは正弦波の要素(値が連続的に変化している部分)と方形波の要素(値が非連続的に変化している部分)が見られるので、この解は正しいと思われる。

- 課題２
  
  課題１で作成したICA.pyを利用して、二人の話者が同時に話したものについて課題１と同じ方法でICAを実行。元のデータの振幅の大きさから、どうしても白色化の際にデータの値が小さくなってしまうので、元のデータの振幅を目安に音声が聞きやすいように振幅を調整。結果得られたデータを5000倍したわけであるが、正直、このやり方は5000という数字に論理的な根拠がないためやや不十分とも思える。しかし、私にはこれよりも良い方法が思いつかない。また、よく聞くと２つに分けたうちのもう片方の音声が小さい音量でまぎれているが、カクテルパーティー効果の再現という意味で考えればこれで十分と判断した。おそらく、同じ作業を何回も繰り返せばより精度が増すと考えられる。

- 課題３
  
  課題２とほぼ同じであるが、白黒画像は情報が２次元配列で表されるため、いったん配列を一次元に変換してからICAを実行、その後再び二次元に戻して画像として出力した。この課題も課題２と同じで、適当な値をデータにかけ、更に負の値を0にしてからキャストした。負の値を0にするのは、そもそも原理的に負の値は出てこないことから、浮動小数点計算の誤差と考えられるからである。ここではとりあえず一番近い正規の値である0を採用した。

## 3.考察

- 収束判定について
  収束判定は繰り返しの結果出てくるベクトルが一致もしくは-1をかけると一致するかどうかで考えた。これは、そもそも符号の取り方さえ統一できていれば、波全体の符号は意味を持たないからである。というのも、信号は複数の信号源の線形結合としてかけるわけだが、ある信号に対して係数aで波を足し合わせることは、それと符号が異なる波を係数(-a)で足し合わせることと同値であるからである。ゆえに、符号の違うw同士は実質的には同じものであり、ゆえに今回の収束判定は問題がないことになる。

- 得られた解の振幅について
  本レポートでは出力する解の振幅は私の感覚的に適していると考えらえるもの、つまりなんとなく決めている。得られた解の比率を求める場合は、本レポートで得られた解に対して、各要素で連立方程式を立て、得られた解をデータにかけるなどすれば求まると思うが、そもそも混合された情報源を分離するのが本レポートの目的であり、その意味では振幅の大きさは問題ではないと考える。それに、得られたデータの比率まで求めたいのならば、数学計算ソフトでフーリエ変換をするほうがよっぽど正確でよいと考える。

## 4.参考文献

- note.nkmk.me(https://note.nkmk.me/)のnumpyに関する記事多数。
- 某エンジニアのお仕事以外のメモ(https://water2litter.net/rum/)のwavfileに関する記事多数。
- プログラミング基礎演習の資料