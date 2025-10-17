import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def random_forest(mlb_df):
    for col in ['WAR','W','L','G','GS','IP','H','R','ER','HR','BB','SO','ERA','WHIP']:
        mlb_df[col].fillna(mlb_df[col].median(), inplace=True)
    mlb_df['Player_Stats_Match'].fillna('', inplace=True)
    agg_dict = {
        'Total Cash': 'mean',
        'WAR': 'sum',
        'IP': 'sum',
        'SO': 'sum',
        'BB': 'sum',
        'ERA': 'mean',
        'WHIP': 'mean'
    }

    mlbg = mlb_df.groupby(['Player', 'Year'], as_index=False).agg(agg_dict)
    mlbg['SO9'] = (mlbg['SO'] / (mlbg['IP']*9)).fillna(0)
    mlbg['Recent'] = (mlbg['Year'] >= 2019).astype(int)

    # 2) consist varies by year 
    mlbg['Year_num'] = mlbg['Year'] - mlbg['Year'].min()   

    # 3) how long the player have already played
    mlbg = mlbg.sort_values(['Player','Year'])
    mlbg['Seasons_Played'] = mlbg.groupby('Player').cumcount() + 1

    # 4) new player
    mlbg['Rookie'] = (mlbg['Seasons_Played'] == 1).astype(int)

    # 5) previous performance is one thing to get high salary

    mlbg['WAR_prev'] = mlbg.groupby('Player')['WAR'].shift(1).fillna(0)
    mlbg['IP_prev']  = mlbg.groupby('Player')['IP'].shift(1).fillna(0)
    mlbg['ERA_prev'] = mlbg.groupby('Player')['ERA'].shift(1).fillna(mlbg['ERA'].median())
    # print(mlbg.shape)
    # mlb.head()


    q3salary = mlbg['Total Cash'].quantile(0.75)
    mlbg['HighPay'] = (mlbg['Total Cash'] > q3salary).astype(int)

    # features
    features = ['WAR', 'ERA', 'IP', 'SO9', 'WHIP','Year_num','Recent','Seasons_Played',
        'WAR_prev','IP_prev','ERA_prev']
    X = mlbg[features].astype('float64')
    y = mlbg['HighPay']

    # split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    #  build and train model
    rf = RandomForestClassifier(
        n_estimators=500,        # tree numbers
        max_depth=20,             # limit the depth avoiding overfitting
        min_samples_split=20,    
        min_samples_leaf=5,      # each leaf at least need 
        class_weight='balanced',
        random_state=42
    )
    rf.fit(X_train, y_train)

    y_prob = rf.predict_proba(X_test)[:,1]
    threshold = 0.6# 門檻從0.5調高到0.6
    y_pred_adj = (y_prob > threshold).astype(int)



    print("Results")
    print("Training Accuracy:", rf.score(X_train, y_train))
    print("Testing Accuracy:", score_with_threshold(rf, X_test, y_test, threshold=0.6))
    print("\nClassification Report:\n", classification_report(y_test, y_pred_adj))

    # Confusion Matrix) 
    plt.figure(figsize=(5,4))
    sns.heatmap(confusion_matrix(y_test, y_pred_adj), annot=True, fmt='d', cmap='Blues', 
                xticklabels=['LowPay','HighPay'], yticklabels=['LowPay','HighPay'])
    plt.title('Confusion Matrix (Random Forest)')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show(block = False)
    import time; time.sleep(0.5)

    # Feature Importance
    importances = pd.DataFrame({
        'Feature': features,
        'Importance': rf.feature_importances_
    }).sort_values(by='Importance', ascending=False)

    plt.figure(figsize=(7,4))
    sns.barplot(x='Importance', y='Feature', data=importances, palette='viridis')
    plt.title('Feature Importance')
    plt.show()

    print("\nTop Feature Importance:\n", importances)

def score_with_threshold(model, X, y, threshold=0.7):
    y_prob = model.predict_proba(X)[:,1]
    y_pred = (y_prob > threshold).astype(int)
    return (y_pred == y).mean()