from impute import *

# Read a tabular dataset on car reviews
df = pd.read_csv("hf://datasets/florentgbelidji/car-reviews/train_car.csv")

# Pre-process data to spare a test set without missing review titles for evaluation purposes
full = df[df['Review_Title'].notna()]
notfull = df[~(df['Review_Title'].notna())]
train, test = train_test_split(full, test_size=df.shape[0]//10,random_state=0)
train = pd.concat([train, notfull])

# Construct the imputation model
imputer = Imputer({'Rating':'int',
                    'Vehicle_Title':'cat',
                    'Review':'text'},
                    'Review_Title')

# Fit the model
imputer.fit(train)

# Impute the review titles in the test set
test['predicted_review_title'] = imputer.predict(test)