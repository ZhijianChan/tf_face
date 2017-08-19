# tf_face:&nbsp;&nbsp;&nbsp;tensoflow codes for face recognition

### Dependencies
* tensorflow 1.1
<br/><br/>
### Core codes
| file | description | execution scripts |
| :---- | :---- | :---- |
| <strong>train_model.py</strong> | CNN training using whole dataset |  run_train.sh |
| <strong>train_model_subset.py</strong> | CNN training using a randomly sampled dataset | run_train_subset.sh |
| <strong>train_fusion.py</strong> | training with multi-patch fusion | run_train_fusion.sh |
| <strong>test_ensemble.py</strong> | testing with model ensembling | run_lfwTest.sh |

<br/><br/>
### Helper codes
| file | usage | execution scripts |
| :---- | :---- | :---- |
| <strong>save_weights.py</strong> | saving trainable parameters as .npy file | run_saveWeights.sh |
| <strong>train_utils.py</strong> | helper code for training | -- |
| <strong>test_utils.py</strong> | helper code for evaluation | -- |
| <strong>test_lfw.py</strong> | helper code for lfw test with pretrained model | run_lfwTest.sh |
| <strong>extract_feature.py</strong> | helper code for feature extraction given a image path list | run_extract.sh |

<br/><br/>
### Why hyper-parameters tunning is important?
![figure1](/pictures/figure1.png)

<br/><br/>
### Some observations of hyper-parameters tunning
#### 1. Learning rate -- dropping lr too quickly may cause early stopping

![figure2](/pictures/figure2.png)

#### 2. Optimizer -- RMSProp vs. Adam

![figure3](/pictures/figure3.png)

#### 3. Dropout -- increase dropout may improve generalization.

![figure4](/pictures/figure4.png)

#### 4. Regularity -- center loss boosts performance

![figure5](/pictures/figure5.png)

#### 5. Dataset -- trained with whole dataset vs. trained with randomly sampled subset
(Maybe modifying randomly sampled method to boostrapping can bring some benefit)

![figure6](/pictures/figure6.png)

<br/><br/>
#### Properly tunned model can push the performance to the limit

![figure7](/pictures/figure7.png)
