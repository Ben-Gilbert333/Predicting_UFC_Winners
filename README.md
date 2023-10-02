![UFC](https://upload.wikimedia.org/wikipedia/commons/thumb/9/92/UFC_Logo.svg/800px-UFC_Logo.svg.png)
# Predicting UFC Winners
## Overview
In this project I analyzed UFC fight data in order to create a model that predicts winners for sports betting purposes. Starting with understanding, exploring, and cleaning the data. Eventually I performed a train/test split in order to start modeling. I tried to create the best model possible. I went through several iterations with each model type using GridSearchCV and RandomizedSearchCV. I settled on a stacking classifier as my best model and used it with my test data. The model results in a profit if a bettor places wagers on its underdog predictions.
## Business Problem
My project is centered around building a predictive model for determining the winners of UFC (Ultimate Fighting Championship) fights, with a specific emphasis on underdog fighters. In sports betting, accurately predicting the outcomes of UFC fights can be a lucrative endeavor. The primary goal of this project is to identify and capitalize on opportunities where underdogs, fighters who are not the favored to win, can be correctly predicted as winners. Underdogs are often considered less likely to win, making their prediction more profitable.

The key performance metric in this project is precision, which aims to reduce false positives. In this scenario underdogs are the positive class. A high-precision model ensures that when it predicts an underdog will win, it is more likely to be correct, reducing the risk involved. This precision-focused approach is important because it minimizes losses from false positives (predicting an underdog will win when they don't) and aims to maximize gains from true positives (predicting an underdog will win, and they do).

The success of my model is not reliant on achieving a high level of accuracy. I am building a predictive model with the intention of only placing bets on underdogs to win, never betting on a favored fighter. In this case, the model only needs to be about 50% precise. This is because of the way the odds are set on fights. The underdog will almost always have plus odds meaning if the bettor wins they profit 100% of their wager at minimum.

If you are new to sports betting or need a better understanding of how the odds work I would highly recommend reading [this article from Forbes](https://www.forbes.com/betting/sports-betting/what-do-sports-betting-odds-mean/#:~:text=Whereas%20negative%20(%2D)%20odds%20tell,%24120%20for%20every%20%24100%20wager.).
## Data Understanding
The dataset I used comes from this page on [Kaggle](https://www.kaggle.com/datasets/mdabbert/ultimate-ufc-dataset?select=ufc-master.csv), it can be downloaded from the link provided or found in the [data folder](https://github.com/Ben-Gilbert333/Predicting_UFC_Winners/tree/main/data) of this GitHub repository. It contains 119 columns and over 5,000 rows of data. Every row in the dataset is a UFC fight. Each row contains various information of the fight including the names of the two fighters, a variety of betting odds, biometrics, win/loss records, and much more. The most important preparation step I took was changing the `winner` column to contain either 'underdog' or 'favored'. The original dataset has the `winner` as either 'blue' or 'red'. In every UFC fight one fighter is in the blue corner and the other is in the red corner. Majority of the time the favored fighter is fighting in the red corner but, this isn't always the case. This is why it was vital to change the `winner` column.

Another important point worth stating clearly was my decision to keep female and male fights together for this model. At first, this may seem like the wrong thing to do. However, I looked at the 13 features I picked for modeling and none of them should have a clear difference separating male fights from female fights. For example, odds makers will set the odds in the same way for male fight as they would for a female fight. The only feature that was concerning in this sense was `reach_dif`. This data takes a measurement of each fighters arm span. Females tend to have much shorter arm spans than men but, it is the difference between the two fighters. Because it is taking the difference I don't think it's a problem because this essentially scales the data.
## Methods
- random forest
- logistic regression
- gradient boosting classifier
- neural network
- stacking classifier
## Results
I chose the stacking classifier as my final model. This used a gradient boosting classifier and a neural network. The neural network was my best model in terms of underfitting/overfitting. It scored about an 82% precision on the training data and 80% on the cross validation. The problem was it very rarely predicted underdogs to win. The gradient boosting classifier predicted underdogs to win often. However, it was overfit with 100% precision on the training data and 48% precision on the cross validation. I used the stacking classifier to try and get the best of both worlds out these two models that were struggling in different ways. The stacking classifier was still overfit with 100% precision on the training data and 65% precision on the cross validation but, it was predicting underdogs to win often. I chose this as my final model because I thought it was the most balanced option.
## Conclusion
My final model had a precision score of 75% on the test data with the default threshold. With the custom threshold the precision score was 48%. 
### Betting Results on Test Data
- Default threshold: 555 dollar profit 
- Custom threshold: 170 dollar profit

This is the profit that would be made if placing a 100 dollar wager on every predicted underdog from the models. These results seem great but, they were disappointing because the model rarely predicts an underdog to win. 
### Next Steps
- Collect Data

    I want to actively collect data on upcoming fights and use my model to predict the winners. Then after the fights are complete I can retrain my model by adding the new data into my training data. I have already began this process, the new data can be found [here](https://github.com/Ben-Gilbert333/Predicting_UFC_Winners/tree/main/data) it is the csv file titled UFC 9-23-23.


- Explore a voting classifier

    I want to model with a voting classifier to use custom weights for the neural network and gradient boosting classifier. Perhaps it will improve the main problem and predict underdogs more often.
    
    
- Feature reduction

    With more time I could explore cutting down on the amount of features used in order to combat overfitting.
## More Information
The notebook can be accessed [here](https://github.com/Ben-Gilbert333/Predicting_UFC_Winners/blob/main/predicting_ufc_winners.ipynb) and presentation slides can be found [here](https://github.com/Ben-Gilbert333/Predicting_UFC_Winners/blob/main/presentation.pdf) for a more in depth look into the project.
## Repository Structure
```
├── Images
    ├── final_model_matrix.png
    ├── grad_boost_matrix.png
    ├── minus_odds.png
    ├── minus_odds_white_labels.png
    ├── neural_network_matrix.png
    ├── plus_odds.png
    └── plus_odds_white_labels.png
├── data
    ├── UFC 9-23-23.csv
    └── ufc-master.csv
├── pickles
    ├── forest_grid_best.pkl
    ├── forest_grid_best2.pkl
    ├── forest_grid_best3.pkl
    ├── forest_grid_best4.pkl
    ├── forest_grid_best5.pkl
    ├── forest_grid_best6.pkl
    ├── gradient_boost_grid_best.pkl
    ├── gradient_boost_grid_best2.pkl
    ├── gradient_boost_grid_best3.pkl
    ├── gradient_boost_grid_best4.pkl
    ├── gradient_boost_grid_best5.pkl
    ├── logreg_grid_best3.pkl
    ├── logreg_grid_best4.pkl
    ├── mlpc_grid_best.pkl
    ├── mlpc_grid_best2.pkl
    ├── mlpc_grid_best3.pkl
    ├── mlpc_grid_best4.pkl
    └── mlpc_grid_best5.pkl
├── scratch_notebooks
    ├── odds_charts.ipynb
    ├── scratch_notebook.ipynb
    ├── scratch_notebook2.ipynb
    ├── scratch_notebook3.ipynb
    └── testing_9-23-23_fights.ipynb
├── .gitignore
├── LICENSE
├── README.md
├── predicting_ufc_winners.ipynb
└── presentation.pdf
```