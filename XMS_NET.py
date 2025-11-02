from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class XMS_NET(BaseEstimator, ClassifierMixin):
    """
    Multi-Level Stacking Classifier
    Level 1: Logistic, RDF, KNN, SVM, GNB
    Level 2: LGB + Level1 meta → final estimator: MLP (RNN)
    """
    def __init__(self,
                 logistic_params=None,
                 knn_params=None,
                 random_state=99):
        self.random_state = random_state
        self.logistic_params = logistic_params
        self.knn_params = knn_params
        
        # Models
        self.LOGISTIC = None
        self.RDF = None
        self.KNN = None
        self.SVM = None
        self.GNB = None
        self.XGB = None
        self.LGB = None
        self.RNN = None
        self.level1 = None
        self.level2 = None
        
    def _build_models(self, X_train, y_train):
        # ---- Logistic Regression (tuning nếu có param grid)
        if isinstance(self.logistic_params, dict):
            grid = GridSearchCV(LogisticRegression(max_iter=1000, random_state=self.random_state),
                                self.logistic_params, cv=5, scoring='accuracy')
            grid.fit(X_train, y_train)
            best_LG = grid.best_params_
        else:
            best_LG = {'max_iter': 1000}
        self.LOGISTIC = LogisticRegression(**best_LG, random_state=self.random_state)

        # ---- Random Forest
        self.RDF = RandomForestClassifier(
            n_estimators=1000,
            max_leaf_nodes=50,
            random_state=self.random_state,
            n_jobs=-1
        )

        # ---- KNN (tuning nếu có param grid)
        if isinstance(self.knn_params, dict):
            grid_knn = GridSearchCV(KNeighborsClassifier(), self.knn_params,
                                    scoring='accuracy', cv=5)
            grid_knn.fit(X_train, y_train)
            best_KNN = grid_knn.best_params_
        else:
            best_KNN = {'n_neighbors': 5}
        self.KNN = KNeighborsClassifier(**best_KNN)

        # ---- SVM
        self.SVM = SVC(kernel='linear', probability=True, random_state=self.random_state)

        # ---- GaussianNB
        self.GNB = GaussianNB()

        # ---- XGBoost (meta1 của Level1)
        self.XGB = XGBClassifier(
            n_estimators=500,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            gamma=1,
            reg_alpha=0.1,
            reg_lambda=0.7,
            random_state=self.random_state,
            use_label_encoder=False,
            eval_metric='logloss'
        )

        # ---- Level1 stacking
        base_model1 = [
            ('LOGISTIC', self.LOGISTIC),
            ('RDF', self.RDF),
            ('KNN', self.KNN),
            ('SVM', self.SVM),
            ('GNB', self.GNB)
        ]
        self.level1 = StackingClassifier(estimators=base_model1,
                                         final_estimator=self.XGB,
                                         cv=5,
                                         n_jobs=-1)

        # ---- Level2 components
        self.LGB = LGBMClassifier(
            objective='binary',
            boosting_type='gbdt',
            num_leaves=20,
            learning_rate=0.005,
            n_estimators=1000,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_samples=10,
            reg_alpha=1.0,
            reg_lambda=1.0,
            random_state=self.random_state,
            n_jobs=-1,
            verbosity=-1
        )

        self.RNN = MLPClassifier(
            random_state=self.random_state,
            max_iter=2000,
            hidden_layer_sizes=(240, 80),
            activation='relu',
            learning_rate='constant',
            learning_rate_init=0.0005,
            alpha=0.0001,
            solver='adam'
        )

        # ---- Level2 stacking (final model)
        level2_estimators = [
            ('LGB', self.LGB),
            ('meta1', self.level1)
        ]
        self.level2 = StackingClassifier(estimators=level2_estimators,
                                         final_estimator=self.RNN,
                                         cv=5,
                                         n_jobs=-1)

    def fit(self, X_train, y_train):
        self._build_models(X_train, y_train)
        self.level2.fit(X_train, y_train)
        return self

    def predict(self, X):
        return self.level2.predict(X)

    def predict_proba(self, X):
        return self.level2.predict_proba(X)

    def evaluate(self, X_test, y_test):
        y_pred = self.predict(X_test)
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred)
        }
        return metrics
