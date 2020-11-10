# GCN_traffic
## Running Environment
python 3.7.4 pytorch 1.4.0 CPU

## data preprocessing
- traffic dataset
The original data is the traffic flow in Sichuan Province, which is confidential. So, I use the PeMS_04 as an alternative.

## 1 Model
- GAT.py

my_net = GAT_model(6,6,1)
### 1.1 Attention Layer
其中GAT使用自定义的attention层 输入为[B,N,in_features] ，输出为[B,N,out_features]
class GraphAttentionLayer(nn.Module):

### 1.2 GAT
A two-layer GAT class

## 2. Model Training
In order to obtain GAT with implicit regularizations and ensure convergence, this paper considers the following three Tricks for two-stage training.
1. Because Adam is not convergent in theory, two-stage training is used. Use Adam to walk to an appropriate parameter area (equivalent to initialization) before solving with SGD

2. The SDD theory with step reduction converges to the optimal point

3. Implicit regularization: For some models (e.g. linear models), SGD can converge to a small norm solution without adding a regular term, which is called implicit regularization. As a Trick here, try SGD as a trainer at the end without proof in order to obtain an implicitly regularized small norm solution.

## 3. Parameter Settings
### 3.1 Random Seed Settings
seed = 1001 
random.seed(seed) 
torch.manual_seed(seed) 
np.random.seed(seed)

### 3.2 Training Settings
- Stage 1 Adam training 5 times
optimizer = torch.optim.Adam(my_net.parameters())
- Stage 2 SGD training 15 times (with step reduction)
optimizer = torch.optim.Adam(my_net.parameters(), lr=0.2, weight_decay=0.005)

## 4. Visualization of traffic flow
### 4.1 Accuracy
#### 4.1.1 Accuracy after Normalization
Test Loss: 0.0846 Mae: 0.0667 Rmse: 0.0920 Mape: 12.92%
#### 4.1.2 Accuracy without Normalization
Mae: 33.38 Rmse: 49.14 Mape: 12.92%

### 4.1.3 Analysis 
Based on the above accuracy indicators, it can be found that the accuracy is good, and the model can currently guide the prediction of traffic flow.However, there is still room for improvement and direction of improvement will be proposed in section 5.

### 4.2 Traffic Visualization
Node 120 is selected for traffic visualization in this paper
#### 4.2.1 Two-week traffic visualization 
We found that the model-based predictions generally conform to the true trend, but because there is too much data to show the details, we need to zoom in on a local time for further analysis.
![figure](https://github.com/UselessOldQian/GCN_traffic/blob/main/Node120%20visualization%20for%202%20weeks.png)

#### 4.2.2 On the first day of traffic visualization
We can see that although the overall effect is okay, there are still two problems.
1. Prediction is too smooth to find high frequency details of real traffic 
2. There is lag
To address these two issues, the fifth section discusses the direction for improvement.
![figure](https://github.com/UselessOldQian/GCN_traffic/blob/main/Node120%20visualization%20for%201%20day.png)

### 4.3 Code Result Display
The original run results are shown here for reference only

Epoch: 0001, Loss: 0.1238, mae: 0.0816, rmse: 0.1108, mape: 0.1625, Time: 4.43 mins
Epoch: 0002, Loss: 0.1078, mae: 0.0759, rmse: 0.1036, mape: 0.1530, Time: 4.45 mins
Epoch: 0003, Loss: 0.1013, mae: 0.0733, rmse: 0.1004, mape: 0.1483, Time: 4.45 mins
Epoch: 0004, Loss: 0.0969, mae: 0.0715, rmse: 0.0982, mape: 0.1450, Time: 4.36 mins
Epoch: 0005, Loss: 0.0937, mae: 0.0702, rmse: 0.0966, mape: 0.1427, Time: 7.22 mins
Epoch: 0006, Loss: 0.0906, mae: 0.0689, rmse: 0.0950, mape: 0.1404, Time: 11.03 mins
Epoch: 0007, Loss: 0.0883, mae: 0.0681, rmse: 0.0938, mape: 0.1387, Time: 5.29 mins
Epoch: 0008, Loss: 0.0869, mae: 0.0675, rmse: 0.0930, mape: 0.1378, Time: 4.10 mins
Epoch: 0009, Loss: 0.0862, mae: 0.0672, rmse: 0.0926, mape: 0.1374, Time: 4.10 mins
Epoch: 0010, Loss: 0.0859, mae: 0.0671, rmse: 0.0925, mape: 0.1373, Time: 4.11 mins
Epoch: 0011, Loss: 0.0857, mae: 0.0670, rmse: 0.0924, mape: 0.1371, Time: 4.11 mins
Epoch: 0012, Loss: 0.0856, mae: 0.0670, rmse: 0.0923, mape: 0.1370, Time: 4.14 mins
Epoch: 0013, Loss: 0.0855, mae: 0.0670, rmse: 0.0923, mape: 0.1369, Time: 4.14 mins
Epoch: 0014, Loss: 0.0854, mae: 0.0669, rmse: 0.0922, mape: 0.1369, Time: 4.19 mins
Epoch: 0015, Loss: 0.0854, mae: 0.0669, rmse: 0.0922, mape: 0.1369, Time: 4.24 mins
Epoch: 0016, Loss: 0.0853, mae: 0.0669, rmse: 0.0922, mape: 0.1369, Time: 4.21 mins
Epoch: 0017, Loss: 0.0853, mae: 0.0669, rmse: 0.0922, mape: 0.1368, Time: 4.21 mins
Epoch: 0018, Loss: 0.0853, mae: 0.0669, rmse: 0.0921, mape: 0.1368, Time: 4.24 mins
Epoch: 0019, Loss: 0.0853, mae: 0.0669, rmse: 0.0922, mape: 0.1368, Time: 4.25 mins
Epoch: 0020, Loss: 0.0852, mae: 0.0668, rmse: 0.0921, mape: 0.1368, Time: 4.26 mins
Test Loss: 0.0846, mae: 0.0667, rmse: 0.0920, mape: 0.1292
Accuracy Indicators Based on Original Values  mae: 33.3855, rmse: 49.1488, mape: 0.1291

The model is saved in "model_GAT_6.pth"

## 5. Improvement direction

For the two problems in the final analysis of Section 4, the following improvements are suggested

1. Boost's method (equivalent to forward step) can be used to capture high frequency details for inadequate capture of traffic, but note that there may be overfitting, so measure carefully.

2. For the hysteresis effect, and considering the periodic effect, it is proposed that multi-interval series can be used to model and analyze, utilize different sequence correlation and capture the long-distance dependence in traffic data, and alleviate the hysteresis effect to some extent.
