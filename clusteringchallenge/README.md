# Howto Bobot this lab

1. You cna modify PipelineWrapper as much as you can, but it cannot use any hardcoded paths within fit/predict functions
2. You can print only to standard error output, **do not print anything to standard output**
3. You can add as many files as you want to **src** directory 
4. Because the training files are large, and Gitlab does not allow files larger than 100MB, you need to read the X_train as follows:
``` python
files = glob("./data/X_train_part_*.csv")
files.sort(key=lambda x: int(re.search(r"(\d+)", x).group()))
X_train = pd.concat((pd.read_csv(f) for f in files), ignore_index=True)
```
5. Have fun :)

# Prizes
Scoreboard: [See here](https://gitlab.com/uj-courses/2024-2025/wpdm/clusteringchallenge/-/wikis/Scoreboard)

| Place | Points |
| ------ | ------ |
| :first_place:  |    5     |
| :second_place: |    3     |
| :third_place:  |    1     |

## More details
- Each repository should contain a `model.py` file located in the `src` directory, as well as a `requirements.txt` file located in the main directory. Additionally, the project can have any number of other files.
- In the `model.py` file, there is a class called `PipelineWrapper` that performs two tasks: `fit` and `predict`. Assume that everything related to training and prediction is encapsulated within this class. Specifically, the parameters for the `fit` and `predict` functions are raw data, which should be prepared and utilized for training and prediction within these functions.
- The results of the project will be visible in the `GRADE.md` file in the student’s repository. However, I reserve the right to exclude information about current results.
- The scoreboard will be accessible on the wiki page for the specific task. For example, for the ClusteringChallenge task, the scoreboard will be on this wiki page: Scoreboard.
- To be eligible in the ranking, you must achieve an Adjusted Rand Index (ARI) score greater than ~ **0.02**. 
- **Note** that scores provided by Bobot are scaled by 100, so in order to pass the assignment, you need to achieve **0.02 * 100 = 2** Bobot point 

## Data Set Characteristics:
- The dataset contains features including numerical and categorical variables.
- The target consists unknown number of clusters between of **2 and 12** that need to be predicted by your model.
- The competition will be won by the team whose clustering solution achieves the highest ARI score against the test set.
- The ARI score is calculated based on the comparison of your predicted cluster labels with the true labels.

### Content of the datasets:
  - **numerical_features**: A mix of continuous and integer values.
  - **categorical_features**: Represented as integers, these should not be one-hot encoded.
  - **target**: The true cluster labels, which are not available during training, but will be used for evaluation against your model’s predictions.

# How your code will be tested?

```python
import model
pw = model.PipelineWrapper()
pw.fit(X_train)
y_pred = pw.predict(X_test)

# Evaluate clustering performance using Adjusted Rand Index (ARI)


from sklearn.metrics import adjusted_rand_score
ari_score = adjusted_rand_score(y_test, y_pred)

# The clustering solution is considered correct if ARI > dynamic threshold set by dummy model.
```