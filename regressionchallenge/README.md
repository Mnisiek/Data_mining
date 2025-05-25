# Howto Bobot this lab

1. You cna modify PipelineWrapper as much as you can, but it cannot use any hardcoded paths within fit/predict functions
2. You can print only to standard error output, **do not print anything to standard output**
3. You can add as many files as you want to **src** directory 
4. Have fun :)

# Prizes
Scoreboard: [See here](https://gitlab.com/uj-courses/2024-2025/wpdm/regressionchallenge/-/wikis/Scoreboard)

| Place | Points |
| ------ | ------ |
| :first_place:  |    5     |
| :second_place: |    3     |
| :third_place:  |    1     |

## More details
- Each repository should contain a model.py file located in the src directory, as well as a requirements.txt file located in the main directory. Additionally, the project can have any number of other files.
- In the model.py file, there is a class called PipelineWrapper that performs two tasks: fit and predict. Assume that everything related to training and prediction is encapsulated within this class. Specifically, the parameters for the fit and predict functions are raw data, which should be prepared and utilized for training and prediction within these functions.
- The results of the project will be visible in the GRADE.md file in the student’s repository. However, I reserve the right to exclude information about current results.
- The scoreboard will be accessible on the wiki page for the specific task. For example, for the RegressionChallenge task, the scoreboard will be on this wiki page: Scoreboard.
- To be eligible in the ranking, you must achieve an r2_score greater than 0.00.

## Data Set Characteristics:
- Target consists of 2 variables: number of regular bikes rent, number of casual bikes shared.
- The competition will win the one who will have the best results on both casual and registered bikes prediction
- The final result for a participant will be calculated as `min(A,B)` where A and B are R2 scores for casual and registered bikes prediction

### Content of the datasets:
  - instant: record index
  - dteday : date
  - season : season (1:winter, 2:spring, 3:summer, 4:fall)
  - yr : year (0: 2011, 1:2012)
  - mnth : month ( 1 to 12)
  - hr : hour (0 to 23)
  - holiday : weather day is holiday or not (extracted from http://dchr.dc.gov/page/holiday-schedule)
  - weekday : day of the week
  - workingday : if day is neither weekend nor holiday is 1, otherwise is 0.
  + weathersit : 
      - 1: Clear, Few clouds, Partly cloudy, Partly cloudy
      - 2: Mist + Cloudy, Mist + Broken clouds, Mist + Few clouds, Mist
      - 3: Light Snow, Light Rain + Thunderstorm + Scattered clouds, Light Rain + Scattered clouds
      - 4: Heavy Rain + Ice Pallets + Thunderstorm + Mist, Snow + Fog
  - temp : Normalized temperature in Celsius. The values are derived via (t-t_min)/(t_max-t_min), t_min=-8, t_max=+39 (only in hourly scale)
  - atemp: Normalized feeling temperature in Celsius. The values are derived via (t-t_min)/(t_max-t_min), t_min=-16, t_max=+50 (only in hourly scale)
  - hum: Normalized humidity. The values are divided to 100 (max)
  - windspeed: Normalized wind speed. The values are divided to 67 (max)
  - casual: count of casual users
  - registered: count of registered users
  - cnt: count of total rental bikes including both casual and registered

# How your code will be tested?

``` python
import model
pw = model.PipelineWrapper()
pw.fit(X_train,y_train)
(y_pred_1, y_pred_2) = pw.predict(X_test)

accuracy1 = r2_score(y_test[:,0], y_pred_1)
accuracy2 = r2_score(y_test[:,1], y_pred_2)

accuracy = min(accuracy1,accuracy2)
```

