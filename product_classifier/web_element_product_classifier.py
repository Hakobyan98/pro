from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from sklearn.ensemble import StackingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import log_loss
from sklearn.decomposition import PCA
from utils import * 

def preprocess_data(file_path):
    df = load_data(file_path)
    df = remove_nan_columns(df)
    df = calculate_position_ratio(df)
    df = remove_useless_columns(df)
    df = encode_categorical_columns(df)

    # Split the data into features and target variables
    y = df['is_shop']
    X = df.drop('is_shop', axis=1)
    X = scale_features(X)
    check_balance(y)

    return X, y

def pca(num_components, X):
    # Perform PCA on the data
    pca_ = PCA(n_components=num_components)
    X_transformed = pca_.fit_transform(X)

    X_reconstructed = pca_.inverse_transform(X_transformed)

    return X_reconstructed

def evaluate_model(model, X_test, y_test):
    # Make predictions on the test data
    y_pred = model.predict(X_test)

    for y in range(len(y_pred)):
        y_pred[y] = 0 if y_pred[y] < 0.5 else 1

    # Evaluate the model's performance
    accuracy = accuracy_score(y_test, y_pred)
    loss = log_loss(y_test, y_pred)

    return accuracy*100, loss

def train_logistic_regression(X_train, y_train, X_test, y_test):
    # Train a logistic regression model and evaluate
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    accuracy, loss = evaluate_model(model, X_test, y_test)
    return accuracy, loss


def train_sequential_model(X_train, y_train, X_test, y_test):
    # Train a sequential model and evaluate
    model = Sequential()
    model.add(Dense(32, activation='relu', input_shape=(X_train.shape[1],)))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, batch_size=8, epochs=20, validation_split=0.2, verbose=0)
    
    accuracy, loss = evaluate_model(model, X_test, y_test)
    
    return accuracy, loss

def train_stacking_classifier(X_train, y_train, X_test, y_test):
    # Train a stacking classifier model and evaluate
    base_models = [
        ('lr', LogisticRegression(max_iter=1000)),
        ('dt', DecisionTreeClassifier(max_depth=10)),
        ('svm', SVC())
    ]
    model = StackingClassifier(
        estimators=base_models,
        final_estimator=LogisticRegression()
    )
    model.fit(X_train, y_train)
    
    accuracy, loss = evaluate_model(model, X_test, y_test)
    return accuracy, loss


def main():

    ## The task is to train a model that classifies whether the web element is product item 
    ## Analyze the results
    
    file_path = 'data_for_classifier.xlsx'
    X, y = preprocess_data(file_path)
    X_ = pca(45, X)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    X_train_, X_test_, y_train_, y_test_ = train_test_split(X_, y, test_size=0.3, random_state=42)

    logistic_accuracy, log_loss = train_logistic_regression(X_train, y_train, X_test, y_test)
    print("Logistic Regression Accuracy: %f, loss is: %f"%(logistic_accuracy, log_loss))
    
    sequential_accuracy, seq_loss = train_sequential_model(X_train, y_train, X_test, y_test)
    print("Sequential Model Accuracy: %f, loss is: %f"%(sequential_accuracy, seq_loss))
    
    # The best model 
    stacking_accuracy, stack_loss = train_stacking_classifier(X_train, y_train, X_test, y_test)
    print("Stacking Classifier Accuracy: %f, loss is: %f"%(stacking_accuracy, stack_loss))

    logistic_accuracy_, log_loss_ = train_logistic_regression(X_train_, y_train_, X_test_, y_test_)
    print("Logistic Regression Accuracy for reconstructed features: %f, loss is: %f"%(logistic_accuracy_, log_loss_))

    ## Using PCA does not imrove the accuracy


if __name__ == "__main__":
    main()
