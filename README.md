# CIFAR-10-API

## 概要

CIFAR-10 画像データセットを用いて作成した画像分類モデルで、APIサーバーを作成しました。

(モデルのレポジトリはこちら : https://github.com/Asato4931/CIFAR-10-Classifier )

## 内容

画像をサーバーにCURLでAPIリクエストを送ると、

- 予測クラス(0~9、対応は以下画像)
- モデルからの信頼度


がレスポンスとして送られます。


### クラス

<img width="705" alt="image" src="https://github.com/Asato4931/CIFAR-10-API/assets/108675293/7c8d545f-f11f-4e95-8caa-8a3602b415b9">


### 結果例

<img width="679" alt="image" src="https://github.com/Asato4931/CIFAR-10-API/assets/108675293/6f349887-2b1b-497b-9004-67fd6a27ec4a">



## 作成手順

APIの基本の解説も含んだ、具体的な作成手順は以下の画像をクリックして参照してください。

<a href = https://qiita.com/asato4931/private/fd3b13ca22ca9cf3b4ea>
  <img width="565" alt="image" src="https://github.com/Asato4931/CIFAR-10-API/assets/108675293/02e50ff0-bc49-423d-a7c9-c9ed9bd592ff">
</a>


