# NLP_電影評論情緒分類

## 研究背景
情緒分析是現今自然語言處理 (NLP) 最重要的應用之一，藉此可以理解人們文字與言論中的情緒，理解評論的真實走向，幫助資料庫建立更加準確的評論系統。

此議題中蒐集了許多簡短的電影文字評論，參賽者需使用現有的評論，訓練人工智慧模型來分析、判斷類似的文字評論帶有正面或是負面的情緒。

## 資料來源
本次資料為電影評論，資料為ID、評論文字及評論情緒0為負面評論;1為正面評論（如下圖）。
![](https://i.imgur.com/BEwbWrY.png)

競賽本身提供29,340筆訓練資料及29,341筆預測資料；另外從huggingface的datasets中IMDB資料集取得了25,000筆訓練資料及25,000筆驗證資料，共50,000筆資料加入訓練資料，以增加訓練資料。

## 研究方法
### 資料預處理
在做資料處理時，有使用過許多方法如刪除stopwords、去除字尾、去除標點符號，但最後進行比較後，發現由於使用BERT進行訓練後，做過多的預處理效果較差，因此最後只做基本的將HTML tag去除即可。預處理完成後會先將資料寫成Json格式，之後重複讀取可減少時間。

### 建置模型
本次使用BertForSequenceClassification訓練模型，並且預先載入bert-base-cased預處理模型。先將所有訓練資料載入後，透過BertTokenizer將訓練資料轉為模型輸入格式以進行訓練。而在模型參數設計上，為減少記憶體使用量，將輸入最長字數設定為256，並參考google-research將batch size設定為8，以避免out of memory。而epoch則同樣參考BERT論文從2、3、4，最後選擇4。

```python=
MAX_LENGTH = 256
BATCH_SIZE = 8
EPOCHS = 4
```
Optimizer部分則是透過多次嘗試並參考論文後進行調整，最後則採用2e-5，但這部分仍舊需要花更多時間進行比較。
```python=
param_optimizer = list(model.named_parameters())
no_decay = ['bias', 'gamma', 'beta']
optimizer_grouped_parameters = [
    {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
     'weight_decay_rate': 0.1},
    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
     'weight_decay_rate': 0.0}
]

optimizer = AdamW(
    optimizer_grouped_parameters,
    lr=2e-5
    eps=1e-8
)
```

## 實驗結果
訓練過程中可明顯發現training loss持續下降，但validation loss卻反而上升，發生overfitting的現象。
| Epoch | Training Loss | Validation Loss | F1       |
|:-----:|:------------- |:--------------- |:-------- |
|   1   | 0.258167      | 0.215078        | 0.938550 |
|   2   | 0.142072      | 0.257556        | 0.948122 |
|   3   | 0.080197      | 0.242050        | 0.957650 |
|   4   | 0.037711      | 0.286389        | 0.959702 |

**而最後則是在競賽Private Leaderboard中得到0.9665206的成績，排名17/121**

## 問題與討論
* 在一開始使用BERT訓練時，在未加入IMDB資料集時大多只能達到0.87的準確率，因此一開始可推斷有訓練資料不足的狀況
* 對於overfitting的狀況之後可考慮調低Learning rate或是增加dropout prob，這部分仍須花時間進行研究

## 參考資料
1. Devlin, Jacob, et al. "Bert: Pre-training of deep bidirectional transformers for language understanding." arXiv preprint arXiv:1810.04805 (2018).
2. [google-research/bert](https://github.com/google-research/bert)