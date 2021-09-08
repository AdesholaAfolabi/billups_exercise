from wrangler.utils import get_attributes
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split


class build_pipeline():
    
    '''
    This class is where the building of the pipeline
    object to be used for modeling takes place
    
    '''
    
    def __init__(self, data):
        
        self.data = data
        self.num_attributes, self.cat_attributes = get_attributes(self.data, ['target', 'customer_id'])
        
        
    def pipeline(self):
        
        """
        
         Function contains the pipeline methods to be used.
         It is broken down into numerical and categorical pipelines
                
        """
        self.num_pipeline = Pipeline(steps= [('imputer', SimpleImputer(strategy='mean')), ('std_scaler', MinMaxScaler())])
        self.cat_pipeline = Pipeline(steps=[('imputer', SimpleImputer(strategy='most_frequent')),
                                       ('one_hot_encoding', OneHotEncoder(handle_unknown = "ignore", sparse = False))])
    
    
    def build_object(self):
        
        """
        
        Function that builds the pipeline and returns the 
        pipeline object and the data to be used for modeling
                
        Args:
            hash_bucket size
        
        Returns:
            pipeline object
            data to be used for training after being transformed by the pipeline
        
        """
        
        self.pipeline()
        self.full_pipeline = ColumnTransformer(
        transformers=[
            ('num', self.num_pipeline, self.num_attributes),
            ('cat', self.cat_pipeline, self.cat_attributes)
        ])
        

        self.y = self.data['target'].copy()
        
        self.X = self.data.drop(['customer_id'], axis=1)
                
        self.X_train, self.X_test, self.y_train, self.y_test = \
        train_test_split(self.X, self.y, test_size=0.2, stratify = self.y)
        
        self.full_pipeline.fit(self.X_train)
        
        self.X_train = self.full_pipeline.transform(self.X_train)
        self.X_test = self.full_pipeline.transform(self.X_test)
        
        print(self.X_train.shape)
        
        return self.X, self.y, self.X_train, self.X_test, self.y_train, self.y_test, self.full_pipeline