# Oil Price Predictor LSTM Model

**Disclaimer: the information in this repository is for educational purposes only**

## Overview
This Flask app employs a long short-term memory (LSTM) recurrent neural network (RNN) model to forecast crude oil prices for the next year. The app fetches historical price data based on the OPEC Crude Oil Basket from the Nasdaq Data Link API and stores it in a SQL database using SQLAlchemy. The LSTM model runs on PyTorch, and the resulting predictions are visualized using Plotly charts. The primary chart highlights key data points such as the highest, lowest, and average oil prices for the predicted period while the secondary chart shows all historical price data from the start of 2003 along with the predicted period. This app may enable users to gain insight into where oil prices could trend going forward.

**LSTM Model Default Parameters:**
- Model trained using mean squared error loss and the Adam optimizer
- Learning rate is set to 0.001.
- Batch size is set to 100.
- Look-back period is set to 100.
- Number of epochs is set to 100.

**Future additions:**
- Make model parameters modifiable
- Add other commodities

**Model output sample (using data from 1/2/2003 - 4/19/2023):**
<img width="1230" alt="image" src="https://user-images.githubusercontent.com/41703555/234064922-626b81b7-ce03-401f-8999-9504376030db.png">

**Epoch breakdown sample:**
```
Epoch [1/100], Train Loss: 0.2310
Epoch [2/100], Train Loss: 0.1906
Epoch [3/100], Train Loss: 0.1445
Epoch [4/100], Train Loss: 0.1160
Epoch [5/100], Train Loss: 0.0988
Epoch [6/100], Train Loss: 0.0870
Epoch [7/100], Train Loss: 0.0784
Epoch [8/100], Train Loss: 0.0718
Epoch [9/100], Train Loss: 0.0667
Epoch [10/100], Train Loss: 0.0625
Epoch [11/100], Train Loss: 0.0591
Epoch [12/100], Train Loss: 0.0562
Epoch [13/100], Train Loss: 0.0537
Epoch [14/100], Train Loss: 0.0516
Epoch [15/100], Train Loss: 0.0498
Epoch [16/100], Train Loss: 0.0481
Epoch [17/100], Train Loss: 0.0467
Epoch [18/100], Train Loss: 0.0454
Epoch [19/100], Train Loss: 0.0442
Epoch [20/100], Train Loss: 0.0432
Epoch [21/100], Train Loss: 0.0422
Epoch [22/100], Train Loss: 0.0413
Epoch [23/100], Train Loss: 0.0405
Epoch [24/100], Train Loss: 0.0397
Epoch [25/100], Train Loss: 0.0390
Epoch [26/100], Train Loss: 0.0384
Epoch [27/100], Train Loss: 0.0378
Epoch [28/100], Train Loss: 0.0372
Epoch [29/100], Train Loss: 0.0367
Epoch [30/100], Train Loss: 0.0362
Epoch [31/100], Train Loss: 0.0357
Epoch [32/100], Train Loss: 0.0353
Epoch [33/100], Train Loss: 0.0349
Epoch [34/100], Train Loss: 0.0345
Epoch [35/100], Train Loss: 0.0341
Epoch [36/100], Train Loss: 0.0337
Epoch [37/100], Train Loss: 0.0334
Epoch [38/100], Train Loss: 0.0331
Epoch [39/100], Train Loss: 0.0327
Epoch [40/100], Train Loss: 0.0325
Epoch [41/100], Train Loss: 0.0322
Epoch [42/100], Train Loss: 0.0319
Epoch [43/100], Train Loss: 0.0316
Epoch [44/100], Train Loss: 0.0314
Epoch [45/100], Train Loss: 0.0312
Epoch [46/100], Train Loss: 0.0309
Epoch [47/100], Train Loss: 0.0307
Epoch [48/100], Train Loss: 0.0305
Epoch [49/100], Train Loss: 0.0303
Epoch [50/100], Train Loss: 0.0301
Epoch [51/100], Train Loss: 0.0299
Epoch [52/100], Train Loss: 0.0297
Epoch [53/100], Train Loss: 0.0295
Epoch [54/100], Train Loss: 0.0293
Epoch [55/100], Train Loss: 0.0292
Epoch [56/100], Train Loss: 0.0290
Epoch [57/100], Train Loss: 0.0289
Epoch [58/100], Train Loss: 0.0287
Epoch [59/100], Train Loss: 0.0286
Epoch [60/100], Train Loss: 0.0284
Epoch [61/100], Train Loss: 0.0283
Epoch [62/100], Train Loss: 0.0281
Epoch [63/100], Train Loss: 0.0280
Epoch [64/100], Train Loss: 0.0279
Epoch [65/100], Train Loss: 0.0277
Epoch [66/100], Train Loss: 0.0276
Epoch [67/100], Train Loss: 0.0275
Epoch [68/100], Train Loss: 0.0274
Epoch [69/100], Train Loss: 0.0273
Epoch [70/100], Train Loss: 0.0271
Epoch [71/100], Train Loss: 0.0270
Epoch [72/100], Train Loss: 0.0269
Epoch [73/100], Train Loss: 0.0268
Epoch [74/100], Train Loss: 0.0267
Epoch [75/100], Train Loss: 0.0266
Epoch [76/100], Train Loss: 0.0265
Epoch [77/100], Train Loss: 0.0264
Epoch [78/100], Train Loss: 0.0263
Epoch [79/100], Train Loss: 0.0263
Epoch [80/100], Train Loss: 0.0262
Epoch [81/100], Train Loss: 0.0261
Epoch [82/100], Train Loss: 0.0260
Epoch [83/100], Train Loss: 0.0259
Epoch [84/100], Train Loss: 0.0258
Epoch [85/100], Train Loss: 0.0258
Epoch [86/100], Train Loss: 0.0257
Epoch [87/100], Train Loss: 0.0256
Epoch [88/100], Train Loss: 0.0255
Epoch [89/100], Train Loss: 0.0255
Epoch [90/100], Train Loss: 0.0254
Epoch [91/100], Train Loss: 0.0253
Epoch [92/100], Train Loss: 0.0253
Epoch [93/100], Train Loss: 0.0252
Epoch [94/100], Train Loss: 0.0251
Epoch [95/100], Train Loss: 0.0250
Epoch [96/100], Train Loss: 0.0250
Epoch [97/100], Train Loss: 0.0249
Epoch [98/100], Train Loss: 0.0248
Epoch [99/100], Train Loss: 0.0247
Epoch [100/100], Train Loss: 0.0247
```
