``` r
library(magrittr)
library(janitor)
```

    ## Warning: package 'janitor' was built under R version 3.4.4

``` r
library(dplyr)
library(tidyr)
library(readr)
library(stringr)
library(forcats)
```

The titanic package contains both the training and test data

``` r
train <- titanic::titanic_train %>% 
  clean_names()
test <- titanic::titanic_test %>% 
  clean_names()
```

A good work through of the analysis is [Trevor Stephen's series of titanic tutorials](http://trevorstephens.com/kaggle-titanic-tutorial/getting-started-with-r/)

Decision Trees
--------------

[Part 3](http://trevorstephens.com/kaggle-titanic-tutorial/r-part-3-decision-trees/) of the tutorial calculates survival using decision trees, using the default control values and titanic variables

``` r
fit <- rpart::rpart(survived ~ pclass + sex + age + sib_sp + parch + fare + embarked,
             data = train,
             method = "class",
             control = rpart::rpart.control(minsplit = 20, cp = 0.01)) %T>% 
  rattle::fancyRpartPlot()
```

![](titanic_survival_machine_learning_gcf_files/figure-markdown_github/unnamed-chunk-3-1.png)

``` r
prediction <- predict(fit, test, type = "class")
```

``` r
# Kaggle submission: score = 0.78468
submit <- test %>%
  mutate(PassengerId = passenger_id,
                Survived = prediction) %>% 
  select(PassengerId, Survived) %T>% 
  write_csv(here::here("survival_from_decision_tree.csv")) 

submit %>% 
  tabyl(Survived) %>% 
  adorn_pct_formatting(digits = 1) 
```

    ## Warning: package 'bindrcpp' was built under R version 3.4.4

    ##  Survived   n percent
    ##         0 288   68.9%
    ##         1 130   31.1%

Feature Engineering
-------------------

Combine the training and test datasets to clean data and extract additional useful variables

``` r
titanic <- bind_rows(train, test) %T>% 
  write_csv(here::here("titanic_data.csv")) 
```

Convert appropriate character fields to factors

``` r
titanic <- titanic %>%
  mutate(
    survived = forcats::as_factor(as.character(survived)),
    pclass = forcats::as_factor(as.character(pclass)),
    sex = forcats::as_factor(sex),
    embarked = forcats::as_factor(embarked)
  )
```

#### Embarked

Two passengers have an unknown embarked port, these are removed from the analysis

``` r
titanic %>% 
  tabyl(embarked) %>% 
  adorn_pct_formatting(digits = 1)
```

    ##  embarked   n percent
    ##         S 914   69.8%
    ##         C 270   20.6%
    ##         Q 123    9.4%
    ##             2    0.2%

``` r
titanic <- titanic %>%
  filter(embarked != "")
```

#### Fare

One third class passenger fare is unknown, replace this with the median price of third class fare ticket

``` r
titanic %>% 
  group_by(pclass) %>% 
  summarise(avg_fare = median(fare, na.rm = TRUE))
```

    ## # A tibble: 3 x 2
    ##   pclass avg_fare
    ##   <fct>     <dbl>
    ## 1 3          8.05
    ## 2 1         60.0 
    ## 3 2         15.0

``` r
titanic <- titanic %>%
  mutate(fare = replace_na(fare, 8.05))
```

Place the fares into four groups - 0-10,10 − 20, 20-50,50+

``` r
titanic <- titanic %>% 
  mutate(fare_bins = cut(fare, breaks = c(0, 10, 20, 50, 513), 
                         include.lowest = TRUE,
                         labels = c("fare_1", "fare_2", "fare_3", "fare_4"),
                         ordered_factor = TRUE))
titanic %>% 
  tabyl(fare_bins) %>% 
  adorn_pct_formatting(digits = 1)
```

    ##  fare_bins   n percent
    ##     fare_1 492   37.6%
    ##     fare_2 261   20.0%
    ##     fare_3 316   24.2%
    ##     fare_4 238   18.2%

#### Age

Age is not know for 263 passengers. Missing ages predicted using either a decision tree or random forest

``` r
# Training dataset
train_age <- titanic %>% 
  filter(!is.na(age))

# Run decision tree on training dataset with known ages
# fit_age <- rpart::rpart(age ~ pclass + sex + sib_sp + parch + fare_bins + embarked,
#                   data = train_age, 
#                   method = "anova")

# Run random forest on training dataset with known ages
set.seed(415)
fit_age <- randomForest::randomForest(age ~ pclass + sex + sib_sp + parch + fare + embarked,
                data = train_age,
                importance = TRUE,
                ntree = 2000) %T>% 
  randomForest::varImpPlot()
```

![](titanic_survival_machine_learning_gcf_files/figure-markdown_github/unnamed-chunk-11-1.png)

``` r
# Update missing ages with predicted ages
test_age <- titanic %>% 
  filter(is.na(age)) %>% 
  mutate(age = predict(fit_age, .))

# Recombine training and test datasets
titanic <- bind_rows(train_age, test_age) %T>% 
  write_csv(here::here("titanic.csv")) 
```

Place the age into five groups - 0-5, 5-10, 10-16, 17-60, 60+

``` r
titanic <- titanic %>% 
  mutate(age_bins = cut(age, breaks = c(0, 5, 10, 16, 60, 80), 
                         include.lowest = TRUE,
                         labels = c("age_1", "age_2", "age_3", "age_4", "age_5"),
                         ordered_factor = TRUE))
titanic %>% 
  tabyl(age_bins) %>% 
  adorn_pct_formatting(digits = 1)
```

    ##  age_bins    n percent
    ##     age_1   56    4.3%
    ##     age_2   30    2.3%
    ##     age_3   63    4.8%
    ##     age_4 1126   86.2%
    ##     age_5   32    2.4%

#### Title

An additional variable to extract is the title from the names. This is taken from Vincent Broute's post [Titanic EDA & predictions attempt](https://www.kaggle.com/neveldo/titanic-eda-predictions-attempt/notebook)

``` r
titanic <- titanic %>%
  mutate(title = str_extract(name, regex("([a-z]+\\.)", ignore_case = TRUE)),
         title = str_replace(title, "\\.", ""),
         title = as_factor(title))

# Group titles into refined groups
titanic <- titanic %>%
  mutate(refined_title = case_when
         (
           title %in% c("Capt", "Don", "Jonkheer", "Rev", "Mr") ~ "title_1",
           title %in% c("Col", "Dr", "Major", "Master") ~ "title_2",
           TRUE ~ "title_3"
          )
      ) %>% 
  mutate(refined_title = as_factor(refined_title))
         
titanic %>% 
  tabyl(title, refined_title) 
```

    ##     title title_1 title_3 title_2
    ##        Mr     757       0       0
    ##       Mrs       0     196       0
    ##      Miss       0     259       0
    ##    Master       0       0      61
    ##       Don       1       0       0
    ##       Rev       8       0       0
    ##        Dr       0       0       8
    ##       Mme       0       1       0
    ##        Ms       0       2       0
    ##     Major       0       0       2
    ##      Lady       0       1       0
    ##       Sir       0       1       0
    ##      Mlle       0       2       0
    ##       Col       0       0       4
    ##      Capt       1       0       0
    ##  Countess       0       1       0
    ##  Jonkheer       1       0       0
    ##      Dona       0       1       0

Training dataset
----------------

``` r
# Training dataset
train_survived <- titanic %>% 
  filter(!is.na(survived))
```

Random Forest
-------------

[Part 5](http://trevorstephens.com/kaggle-titanic-tutorial/r-part-5-random-forests/) of the tutorial calculates survival using random forest method

Random Forest method does not allow for missing values and requires factors for Survived and discrete variables

``` r
# Run random forest on training dataset with known survival
set.seed(415)
fit_survived <- randomForest::randomForest(survived ~ pclass + sex + age_bins + sib_sp + parch + fare_bins + embarked + refined_title,
                data = train_survived,
                importance = TRUE,
                ntree = 2000) %T>% 
  randomForest::varImpPlot()
```

![](titanic_survival_machine_learning_gcf_files/figure-markdown_github/unnamed-chunk-15-1.png)

``` r
# Update missing survival with predicted survival
test_survived <- titanic %>% 
  filter(is.na(survived)) %>% 
  mutate(survived = predict(fit_survived, .))

# Recombine training and test datasets
titanic_rf <- bind_rows(train_survived, test_survived) %T>% 
  write_csv(here::here("titanic_rf.csv")) 
```

Conditional inference trees
---------------------------

``` r
set.seed(415)

# Run Conditional inference trees on training dataset with known survival
fit_survived <- party::cforest(survived ~ pclass + sex + age + sib_sp + parch + fare + embarked + title,
                      data = train_survived,
                      controls = party::cforest_unbiased(ntree = 2000, mtry = 3))
                      
# Update missing survival with predicted survival
test_survived <- titanic %>% 
  filter(is.na(survived)) %>% 
  mutate(survived = predict(fit_survived, ., OOB = TRUE, type = "response"))

# Recombine training and test datasets
titanic_ci <- bind_rows(train_survived, test_survived) %T>% 
  write_csv(here::here("titanic_ci.csv")) 
```

``` r
# Kaggle submission: score = 0.80861
submit <- test_survived %>%
  mutate(PassengerId = passenger_id,
                Survived = survived) %>% 
  select(PassengerId, Survived) %T>% 
  write_csv(here::here("kaggle_submission.csv")) 

submit %>% 
  tabyl(Survived) %>% 
  adorn_pct_formatting(digits = 1) 
```

    ##  Survived   n percent
    ##         0 275   65.8%
    ##         1 143   34.2%
