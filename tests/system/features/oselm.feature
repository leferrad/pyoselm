
Feature: Fit and validate an OSELMRegressor model with input data
  As a data scientist,
  I want to fit and validate an OSELMRegressor with input data
  in order to get a regression model with proper results

  @regression @oselm
  Scenario: Fit an OSELMRegressor with toy data (easy)
    Given the dataset 'boston'
    And a pre-processing pipeline
    And an OSELMRegressor model
    When I fit the pipeline and the model
    Then I compute the score in train and test sets 
    And the results are very good

  @regression @oselm
  Scenario: Fit an OSELMRegressor with real data (hard)
    Given the dataset 'california'
    And a pre-processing pipeline
    And an OSELMRegressor model
    When I fit the pipeline and the model
    Then I compute the score in train and test sets 
    And the results are good enough

  @classification @oselm
  Scenario: Fit an OSELMClassifier with toy data (easy)
    Given the dataset 'iris'
    And a pre-processing pipeline
    And an OSELMClassifier model
    When I fit the pipeline and the model
    Then I compute the score in train and test sets 
    And the results are very good

  @classification @oselm @expensive
  Scenario: Fit an OSELMClassifier with real data (hard)
    Given the dataset 'covertype'
    And a pre-processing pipeline
    And an OSELMClassifier model
    When I fit the pipeline and the model
    Then I compute the score in train and test sets
    And the results are good enough

  @regression @oselm @expensive
  Scenario: Fit an OSELMRegressor with toy data in online fashion, row by row
    Given the dataset 'boston'
    And a pre-processing pipeline
    And an OSELMRegressor model
    When I fit the pipeline and the model in online fashion, row by row,
    Then I compute the score in train and test sets
    And the results are good enough

  @regression @oselm  @expensive
  Scenario: Fit an OSELMRegressor with toy data in online fashion, chunk by chunk
    Given the dataset 'boston'
    And a pre-processing pipeline
    And an OSELMRegressor model
    When I fit the pipeline and the model in online fashion, chunk by chunk,
    Then I compute the score in train and test sets
    And the results are good enough


