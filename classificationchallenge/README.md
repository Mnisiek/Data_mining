# Howto Bobot this lab

1. You cna modify PipelineWrapper as much as you can, but it cannot use any hardcoded paths within fit/predict functions
2. You can print only to standard error output, **do not print anything to standard output**
3. You can add as many files as you want to **src** directory 
4. Have fun :)

# Prizes
Scoreboard: [See here](https://gitlab.com/uj-courses/2024-2025/wpdm/classificationchallenge/-/wikis/Scoreboard)

| Place | Points |
| ------ | ------ |
| :first_place:  |    5     |
| :second_place: |    3     |
| :third_place:  |    1     |

## More details
- Each repository should contain a model.py file located in the src directory, as well as a requirements.txt file located in the main directory. Additionally, the project can have any number of other files.
- In the model.py file, there is a class called PipelineWrapper that performs two tasks: fit and predict. Assume that everything related to training and prediction is encapsulated within this class. Specifically, the parameters for the fit and predict functions are raw data, which should be prepared and utilized for training and prediction within these functions.
- The results of the project will be visible in the GRADE.md file in the student’s repository. However, I reserve the right to exclude information about current results.
- The scoreboard will be accessible on the wiki page for the specific task. For example, for the ClassificationChallenge task, the scoreboard will be on this wiki page: [Scoreboard](https://gitlab.com/uj-courses/2024-2025/wpdm/classificationchallenge/-/wikis/Scoreboard).
- To be eligible in the ranking, you must achieve an f1 score greater than 0.00.

## Data Set Characteristics:
- Target consists of 1 variable representing a sentiment of an article about bitcoin economy. It can be 0,1 or 2 representing negative, neutral and positive sentiment.
- The competition will win the one who will have the best results on predicting correct sentiment
- The final result for a participant will be calculated as `F1 score` with macro average.

### Content of the datasets:
  - text - text about some aspect of cryptocurrency world
  - sentiment - the sentiment of the text on a 0-1-2 scale.

# How your code will be tested?

``` python
import model
pw = model.PipelineWrapper()
pw.fit(X_train,y_train)
y_pred_1= pw.predict(X_test)

score = f1_score(y_test, y_pred_1,average='macro')

```



