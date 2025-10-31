## ADDED Requirements
### Requirement: Spam Classification Baseline
The system SHALL provide a baseline supervised model to classify messages as spam or ham using logistic regression.

#### Scenario: Train baseline model
- **WHEN** the training script is executed with the provided dataset URL
- **THEN** the system SHALL download and preprocess the data, train a logistic regression model with TF-IDF features, persist the trained model to disk, and output evaluation metrics

#### Scenario: Predict using trained model
- **WHEN** a user runs the prediction CLI with a message text
- **THEN** the system SHALL load the persisted model and return a predicted label (`spam` or `ham`) along with a confidence score
