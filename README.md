# SCAN: A Spatial Context Attentive Network for Joint Multi-Agent Intent Prediction

Code for the paper

[SCAN: A Spatial Context Attentive Network for Joint Multi-Agent Intent Prediction](https://arxiv.org/abs/2102.00109)
Jasmine Sekhon, Cody Fleming 
Accepted at AAAI-2021 

**SCAN** is a **S**patial **C**ontext **A**ttentive **N**etwork that can jointly predict trajectories for all pedestrians in a scene over a future time window by attending to spatial contexts experienced by them individually over an observed time window. 

## Model Architecture

Our model contains an LSTM-based Encoder-Decoder Framework that takes as input observed trajectories for all pedestrians in the frame and <em> jointly predicts </em> future trajectories for all the pedestrians in the given frame. To account for spatial influences of spatially close pedestrians on each other, our model uses a <em> Spatial Attention Mechanism </em> that infers and incorporates perceived spatial contexts into each pedestrian's LSTM's knowledge. In the decoder, our model additionally uses a temporal attention mechanism to attend to the observed spatial contexts for each pedestrian, to enable the model to learn how to navigate by learning from previously encountered spatial situations. 

<img src = https://github.com/coordinated-systems-lab/AAAI-21-Submission/blob/master/model.png width="1000" height="500">

## Example Predictions

### Socially Acceptable Future Trajectories 
Our spatial attention mechanism is able to learn and account for the influence of neighboring pedestrians' observed and future trajectories on a pedestrian. Therefore, our model's predictions reflect behavior that respects social navigation norms, such as, avoiding collision, yielding right-of-way and is also able to exhibit complex social behavior such as walking in groups/pairs, groups avoiding collisions with other groups and so on. Below we show some examples of such behavior:

<img src = https://github.com/coordinated-systems-lab/AAAI-21-Submission/blob/master/img/group.gif width="300" height="250"> <img src = https://github.com/coordinated-systems-lab/AAAI-21-Submission/blob/master/img/group_group_2.gif width="300" height="250"> <img src = https://github.com/coordinated-systems-lab/AAAI-21-Submission/blob/master/img/collision_avoidance.gif width="300" height="250">


### Multiple Socially Plausible Predictions
Human motion is multimodal, and often there is no single correct future trajectories. Given an observed trajectory and spatial context, a pedestrian may follow several different trajectories in the future. Taking this uncertain nature of pedestrian motion into account, we also propose **GenerativeSCAN** which is a GAN-based SCAN framework, capable of generating multiple socially feasible future trajectories for all pedestrians in the frame. Below we show an example of how three pedestrians are predicted to engage in collision avoidance behavior differently in six different predictions by **GenerativeSCAN**. Because all the predicted trajectories avoid collisions, they are <em> socially plausible </em>. The sample is from ZARA1 test dataset plotted for <em>k=10</em> and \lambda=0.001. 

  <img src = https://github.com/coordinated-systems-lab/AAAI-21-Submission/blob/master/img/pred_1.gif width="300" height="250"> <img src = https://github.com/coordinated-systems-lab/AAAI-21-Submission/blob/master/img/pred_2.gif width="300" height="250"> 
  <img src = https://github.com/coordinated-systems-lab/AAAI-21-Submission/blob/master/img/pred_3.gif width="300" height="250"> <img src = https://github.com/coordinated-systems-lab/AAAI-21-Submission/blob/master/img/pred_4.gif width="300" height="250">
  <img src = https://github.com/coordinated-systems-lab/AAAI-21-Submission/blob/master/img/pred_5.gif width="300" height="250"> <img src = https://github.com/coordinated-systems-lab/AAAI-21-Submission/blob/master/img/pred_6.gif width="300" height="250"> 

  
## Training Details

We train and evaluate our models on five publicly available datasets: ETH, HOTEL, UNIV, ZARA1, ZARA2. We follow a leave-one-out process where we train on four of the five models and test on the fifth. The exact training, validation, test datasets we use are in directory data/ . For each pedestrian in a given frame, our model observes the trajectory for 8 time steps (3.2 seconds) and predicts intent over future 8 time steps (3.2 seconds) jointly for all pedestrians in the scene.

There are several available models to choose from, main variants being:
1. `vanilla`, which is a vanilla LSTM-based autoencoder,
2. `temporal`, which is an LSTM-based autoencoder with temporal attention in the decoder, 
3. `spatial`, which is referred to in the paper as **vanillaSCAN**, and is an LSTM-based autoencoder with spatial attention mechanism, 
4. `spatial_temporal`, which is our proposed model **SCAN**, 
5. `generative_spatial_temporal`, which is **GenerativeSCAN**, a GAN-based SCAN capable of predicting multiple socially plausible trajectories. 

<em> in progress: SCAN with scene context trained to extract scene relevant features from the static scene image pertaining to a dataset, using pretrained resnet-18 model </em>

To train **SCAN**, i.e., the deterministic model with our chosen hyperparameters, simply edit `--dset_name` and `model_type` arguments in the `scripts/script_train_deterministic.sh` script. 
All other arguments are specified and explained in `arguments.py`. 

To train **GenerativeSCAN** with our chosen hyperparameters, similarly edit `--dset_name` and `model_type` arguments in the `scripts/script_train_generative.sh` script. 
All other arguments are specified and explained in `arguments.py`.

To evaluate trained models for **SCAN**, i.e., the deterministic model with our chosen hyperparameters on all of the datasets, run

```
sh scripts/script_evaluate_deterministic.sh 
```





