# CIFAR-10-API

## 概要

CIFAR-10 画像データセットを用いて作成した画像分類モデル※で、APIサーバーを作成しました。
(※モデルのレポジトリはこちら : https://github.com/Asato4931/CIFAR-10-Classifier )

## 内容

画像をサーバーにCURLでAPIリクエストを送ると、

・予測クラス(0~9、対応は以下画像)
・モデルからの信頼度

がレスポンスとして送られます。


https://camo.qiitausercontent.com/76ebdabec458aa46fb7184f613822d0e9ad86044/68747470733a2f2f71696974612d696d6167652d73746f72652e73332e61702d6e6f727468656173742d312e616d617a6f6e6177732e636f6d2f302f333533343535312f37353463653961312d393266312d353663382d666634612d6635383766343934323362342e706e67![image](https://github.com/Asato4931/CIFAR-10-API/assets/108675293/1534b6f6-62ff-4f44-b3d6-39c8303a1eed)



## 作成手順

具体的な作成手順は以下を参照ください。

https://qiita.com/asato4931/private/fd3b13ca22ca9cf3b4ea
