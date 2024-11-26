# DM-2024 Lab2 HW
code by 111060026 daisyliu0225
name: 劉孟瑋
Student ID: 111060026
github ID: daisyliu0225
Kaggle name: daisyliu0225

---
# Adapted Versions
## Code 1: BOW + Deep learning
The code in the repo is: version2_BOW
### Versions
|Version|Max Features|Train Data Sample|epoch|batch size|    Result    |
|-------|------------|-----------------|-----|----------|--------------|
|Original|    500    |      0.3        |25   |32        |    0.36322   |
|Version 1|    500   |      0.3        |25   |40        |    0.36322   |
|Version 2|    500   |      0.3        |40   |32        |    0.36322   |
|Version 3|    1000  |      0.3        |40   |512       |    0.37987   |
|Version 4|    2000  |      0.3        |40   |512       |    0.37959   |
|Version 5|    1800  |      0.3        |40   |512       |    0.38144   |
|Version 6|    1800  |       1         |40   |512       |    0.41604   |

A few conclusions:
1. epoch and batch size does not significant affect the result
2. Train data sample can affect the result(I did not test it on kaggle), but I had a small "accuracy test label" that helps me test what is going on in my code. 
![acc test label](./for_readme/accuracy.png)
3. The number of max features has a limit and will decrease if exceed the limit. The limit in this data is around 1800.


---
# Discarded Versions
## Code 1: Decision Tree
The code in the repo is: version1_decision_tree<br>
Discarded reason: Training the decision tree is too long and the accuracy is too low. Only around 0.3.<br>
## Code 2: BERT
The code in the repo is: version3_bert<br>
It takes too long to predict. According to GPT, it may take up to days if using a CPU and hours using GPU.
## Code 3: n_grams + Deep learning
The code in the repo is: version4_n_grams<br>
Discarded reason: Training n-grams is too long and the accuracy is too low. It is about 0.30501.

