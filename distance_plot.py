# from PIL import Image
# #import optuna
# import numpy as np
# from numpy import asarray
# import os
# from matplotlib import pyplot as plt
# from sklearn.preprocessing import StandardScaler
# from scipy.linalg import eigh
# from sklearn.model_selection import cross_val_score
# from sklearn.metrics import accuracy_score
# from sklearn.model_selection import train_test_split
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.metrics import confusion_matrix
# from sklearn.metrics import classification_report
# from sklearn.decomposition import PCA
# from sklearn.ensemble import RandomForestClassifier
# #from sklearn.lda import LDA
# from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
# from sklearn.model_selection import GridSearchCV
# from sklearn.metrics import classification_report
# from sklearn.svm import SVC
# from sklearn.decomposition import PCA as RandomizedPCA
# from scipy.spatial import distance

# class FaceRecognition:
#     def get_train_test_splits(self, folder, train_size=8):
#         image_folders = []
#         for f in os.listdir(folder):
#             if f == 'README':
#                 continue
#             image_folders.append(f)

#         x_first, y_first = True, True
#         y_train, y_test = [], []

#         for f in image_folders:
#             loc = folder + '/' + f
#             count = 0
#             folnum = int(f[1:])
#             for file in os.listdir(loc):
#                 file_loc = loc + '/' + file
#                 image = Image.open(file_loc)
#                 pixels = asarray(image)
#                 pixels = np.reshape(pixels, [1, pixels.shape[0]*pixels.shape[1]])

#                 if count < train_size:
#                     if x_first:
#                         X_train = pixels
#                         x_first = False
#                     else:
#                         X_train = np.vstack([X_train, pixels])
#                     y_train.append(folnum)
#                 else:
#                     if y_first:
#                         X_test = pixels
#                         y_first = False
#                     else:
#                         X_test = np.vstack([X_test, pixels])
#                     y_test.append(folnum)
#                 count += 1
#         return X_train, X_test, y_train, y_test

#     def __init__(self, folder=r'dataset', train_size=8):
#         X_train, X_test, y_train, y_test = self.get_train_test_splits(folder=folder, train_size=train_size)
#         self.X_train = X_train
#         self.X_test = X_test
#         self.y_train = y_train
#         self.y_test = y_test
#         self.covariance_matrix = None
#         self.eigenval = None
#         self.eigenvec = None
#         self.z = None

#     def feature_extraction(self, n_components=150, verbose=False):
#         if verbose:
#             print('Before feature extraction: ',
#                   self.X_train.shape, self.X_test.shape)
#             self.plot_cumulative_variance(PCA().fit(self.X_train))
#         pca = PCA(n_components=n_components)
#         self.X_train = pca.fit_transform(self.X_train)
#         self.X_test = pca.transform(self.X_test)
#         self.covariance_matrix = np.cov(self.X_train)
#         self.eigenval, self.eigenvec = np.linalg.eigh(self.covariance_matrix)
#         explained_variances = pca.explained_variance_ratio_
#         feature_weights = np.sum((pca.components_ ** 2) * explained_variances.reshape(-1, 1), axis=0)
#         self.z = feature_weights
#         if verbose:
#             # print('Variance explained by each component: ', pca.explained_variance_ratio_)
#             print('Total variance explained: ', np.sum(pca.explained_variance_ratio_))
#             self.plot_cumulative_variance(pca)

#     def plot_cumulative_variance(self, pca):
#         plt.figure(figsize=(10, 7))
#         plt.ylim(0.0, 1.1)
#         plt.plot(np.cumsum(pca.explained_variance_ratio_), linewidth=3)
#         plt.xlabel('number of components', fontsize=21)
#         plt.ylabel('cumulative explained variance', fontsize=21)
#         plt.title('Scree Plot using PCA', fontsize=24)
#         plt.rc('font', size=16)
#         plt.grid()
#         plt.show()

#     def model_selection(self, model='Random Forest Optuna', param=None, verbose=True):
#         if model == 'Random Forest':
#             acc = self.random_forest(param, verbose)
#             print('Accuracy of Random Forest: ', acc)

#         elif model == 'Random Forest Optuna':
#             param = self.random_forest_optuna()
#             print('Best parameters: ', param)
#             acc = self.random_forest(param, verbose)
#             print('Accuracy of Random Forest Optuna: ', acc)

#         elif model == 'Distance Based':
#             distance = input('Choose distance: ')
#             plot = input("Plot or not? ")
#             acc = self.distance_based(distance, plot=plot)
#             print('Accuracy of Distance Based: ', acc)

#         elif model == 'KNN':
#             acc = self.knn(verbose=verbose)
#             print('Accuracy of KNN: ', acc)

#         elif model == 'SVM':
#             acc = self.svm(param, verbose)
#             print('Accuracy of SVM: ', acc)

#         elif model == 'SVM Optuna':
#             param = self.svm_optuna()
#             print('Best parameters: ', param)
#             acc = self.svm(param, verbose)
#             print('Accuracy of SVM Optuna: ', acc)

#     def random_forest(self, param, verbose):
#         rf = RandomForestClassifier(**param)
#         rf.fit(self.X_train, self.y_train)
#         y_pred = rf.predict(self.X_test)
#         if verbose:
#             print(classification_report(self.y_test, y_pred, zero_division=0))
#         return accuracy_score(self.y_test, y_pred)

#     def random_forest_optuna(self, n_trials=100):
#         def objective(trial):
#             n_estimators = trial.suggest_int('n_estimators', 200, 2000)
#             max_depth = trial.suggest_int('max_depth', 10, 100)
#             min_samples_split = trial.suggest_int('min_samples_split', 2, 10)
#             min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 10)
#             bootstrap = trial.suggest_categorical('bootstrap', [True, False])
#             clf = RandomForestClassifier(
#                 n_estimators=n_estimators, max_depth=max_depth, min_samples_split=min_samples_split,
#                 min_samples_leaf=min_samples_leaf, bootstrap=bootstrap, n_jobs=-1)
#             return cross_val_score(clf, self.X_train, self.y_train, n_jobs=-1, cv=3).mean()

#         study = optuna.create_study(direction='maximize')
#         study.optimize(objective, n_trials=n_trials)
#         print('Number of finished trials:', len(study.trials))
#         print('Best trial:', study.best_trial.params)

#         return study.best_trial.params

#     def svm(self, param, verbose):
#         svm = SVC(**param)
#         svm.fit(self.X_train, self.y_train)
#         y_pred = svm.predict(self.X_test)
#         if verbose:
#             print(classification_report(self.y_test, y_pred, zero_division=0))
#         return accuracy_score(self.y_test, y_pred)

#     # def svm_optuna(self, n_trials=20):
#     #     def objective(trial):
#     #         C = trial.suggest_float('C', 1e-2, 1e2, log=True)
#     #         gamma = trial.suggest_float('gamma', 1e-5, 1e1, log=True)
#     #         kernel = trial.suggest_categorical(
#     #             'kernel', ['rbf', 'linear', 'poly', 'sigmoid'])
#     #         clf = SVC(C=C, gamma=gamma, kernel=kernel)
#     #         return cross_val_score(clf, self.X_train, self.y_train, n_jobs=-1, cv=3).mean()

#     #     study = optuna.create_study(direction='maximize')
#     #     study.optimize(objective, n_trials=n_trials)
#     #     print('Number of finished trials:', len(study.trials))
#     #     print('Best trial:', study.best_trial.params)

#     #     return study.best_trial.params

#     def knn(self, n_neighbors=5, verbose=False):
#         knn = KNeighborsClassifier(n_neighbors=n_neighbors)
#         knn.fit(self.X_train, self.y_train)
#         y_pred = knn.predict(self.X_test)
#         if verbose:
#             print(classification_report(self.y_test, y_pred, zero_division=0))
#         return accuracy_score(self.y_test, y_pred)

#     def distance_based(self, distance='euclidean', return_pairs=False, plot="no"):
#         if plot.lower()=='yes':
#             self.visualize_correct_top_predictions(distance = distance)
#             return self.visualize_wrong_predictions(distance=distance)
#         else:
#             if distance == 'euclidean':
#                 return self.euclidean_distance(return_pairs=return_pairs)
#             elif distance == 'cosine':
#                 return self.cosine_similarity(return_pairs=return_pairs)
#             elif distance == 'minkowski':
#                 return self.minkowski_distance(return_pairs=return_pairs)
#             elif distance == 'mahala':
#                 return self.simplified_mahala(return_pairs=return_pairs)
#             elif distance == 'angle':
#                 version = input('Weighted or normal angle base?')
#                 return self.angle(version = version, return_pairs=return_pairs)
#             elif distance == 'sse':
#                 version = input('normal SSE or modified SSE? ')
#                 return self.sse(version = version, return_pairs=return_pairs)
        
#     def angle(self, version = 'normal', return_pairs=False):
#         if version == 'normal':
#             return self.cosine_similarity(return_pairs=return_pairs)
#         elif version == 'weighted':
#             return self.angle_base(return_pairs=return_pairs)


#     def cosine_similarity(self,return_pairs = False):
#         y_pred = []
#         pairs = []
#         distances = []
#         for i in range(self.X_test.shape[0]):
#             sim = []
#             for j in range(self.X_train.shape[0]):
#                 sim.append(np.dot(self.X_test[i], self.X_train[j])/(np.linalg.norm(self.X_test[i])*np.linalg.norm(self.X_train[j])))
#             y_pred.append(self.y_train[np.argmax(sim)])
#             pairs.append((i, np.argmax(sim)))
#             distances.append(np.max(sim))

#         if return_pairs:
#             return distances, pairs, y_pred
#         else:
#             return accuracy_score(self.y_test, y_pred)

#     def minkowski_distance(self,return_pairs = False):
#         y_pred = []
#         pairs = []
#         distances = []
#         for i in range(self.X_test.shape[0]):
#             dist = []
#             for j in range(self.X_train.shape[0]):
#                 dist.append(np.sum(np.abs(self.X_test[i]-self.X_train[j])**3)**(1/3))
#             y_pred.append(self.y_train[np.argmin(dist)])
#             pairs.append((i, np.argmin(dist)))
#             distances.append(np.min(dist))

#         if return_pairs:
#             return distances, pairs, y_pred
#         else:
#             return accuracy_score(self.y_test, y_pred)
    
#     def angle_base(self, return_pairs = False):
#         z = self.z
#         y_pred = []
#         pairs = []
#         distances = []
#         for i in range(self.X_test.shape[0]):
#             dis = []
#             for j in range(self.X_train.shape[0]):
#                 dis.append(np.dot(z[i],np.dot(self.X_test[i], self.X_train[j]))/(np.linalg.norm(self.X_test[i])*np.linalg.norm(self.X_train[j])))
#             y_pred.append(self.y_train[np.argmax(dis)])
#             pairs.append((i, np.argmax(dis)))
#             distances.append(np.max(dis))

#         if return_pairs:
#             return distances, pairs, y_pred
#         else:
#             return accuracy_score(self.y_test, y_pred)


#     def simplified_mahala(self, return_pairs = False):
#         y_pred = []
#         pairs = []
#         distances = []
#         z_i = self.z
#         for i in range(self.X_test.shape[0]):
#             distance = []
#             for j in range(self.X_train.shape[0]):
#                 distance.append((np.dot(z_i[j],np.dot(self.X_test[i], self.X_train[j]))))
#             y_pred.append(self.y_train[np.argmax(distance)])
#             pairs.append((i, np.argmax(distance)))
#             distances.append(np.max(distance))

#         if return_pairs:
#             return distances, pairs, y_pred
#         else:
#             return accuracy_score(self.y_test, y_pred)

    
#     def sse(self,version, return_pairs = False):
#         y_pred = []
#         pairs = []
#         distances = []
#         for i in range(self.X_test.shape[0]):
#             dist = []
#             for j in range(self.X_train.shape[0]):
#                 if version == 'normal':
#                     dist.append(np.sum((self.X_test[i] - self.X_train[j])**2))
#                 elif version == 'modified':
#                     dist.append(np.sum((self.X_test[i] - self.X_train[j])**2)/((np.sum((self.X_test[i])**2)) + np.sum((self.X_train[j])**2)))
#             y_pred.append(self.y_train[np.argmin(dist)])
#             pairs.append((i, np.argmin(dist)))
#             distances.append(np.min(dist))

#         if return_pairs:
#             return distances, pairs, y_pred
#         else:
#             return accuracy_score(self.y_test, y_pred)            
    
    
#     def euclidean_distance(self, return_pairs=False):
#         y_pred = []
#         pairs = []
#         distances = []
#         for i in range(self.X_test.shape[0]):
#             dist = []
#             for j in range(self.X_train.shape[0]):
#                 dist.append(np.linalg.norm(self.X_test[i] - self.X_train[j]))
#             y_pred.append(self.y_train[np.argmin(dist)])
#             pairs.append((i, np.argmin(dist)))
#             distances.append(np.min(dist))

#         if return_pairs:
#             return distances, pairs, y_pred
#         else:
#             return accuracy_score(self.y_test, y_pred)

    
#     def visualize_wrong_predictions(self, distance='euclidean'):
#         distance, pairs, y_pred = self.distance_based(distance=distance, return_pairs=True)

#         wrong_predictions = [(i, pairs[i][1]) for i in range(len(pairs)) if self.y_test[i] != y_pred[i]]
#         print(f'Number of mismatched cases: {len(wrong_predictions)}')

#         plt.figure(figsize=(15, 5 * len(wrong_predictions)))

#         for i, (test_index, train_index) in enumerate(wrong_predictions, start=1):
#             test_folder = f"s{self.y_test[test_index]}"
#             train_folder = f"s{self.y_train[train_index]}"

#             test_image_path = os.path.join('dataset', test_folder, f"{test_index % 10 + 1}.pgm")
#             train_image_path = os.path.join('dataset', train_folder, f"{train_index % 10 + 1}.pgm")

#             test_image = Image.open(test_image_path)
#             train_image = Image.open(train_image_path)

#             plt.subplot(len(wrong_predictions), 2, i * 2 - 1)
#             plt.imshow(test_image, cmap='gray')
#             plt.title(f'Test Image (Actual: {test_folder}, Predicted: {self.y_train[train_index]})')
#             plt.axis('off')

#             plt.subplot(len(wrong_predictions), 2, i * 2)
#             plt.imshow(train_image, cmap='gray')
#             plt.title(f'Predicted Train Image (Folder: {train_folder})')
#             plt.axis('off')

#         plt.tight_layout()
#         plt.show()

#         return accuracy_score(self.y_test, y_pred)
    


#     def visualize_correct_top_predictions(self,distance = "euclidean"):
#         distances, pairs, y_pred = self.distance_based(distance=distance, return_pairs=True)

#         true_predictions = [(i, pairs[i][1]) for i in range(len(pairs)) if self.y_test[i] == y_pred[i]]

#         if len(true_predictions) > 0:
#             # Find the index corresponding to the smallest distance among true positive predictions
#             min_distance_index = min(true_predictions, key=lambda x: distances[x[0]])

#             min_distance_index = min_distance_index[0]
#             min_distance_pair = pairs[min_distance_index]
#             min_distance_value = distances[min_distance_index]

#             print(f'Top matching pair for {distance} (Lowest Distance: {min_distance_value}):')

#             plt.figure(figsize=(10, 5))

#             test_index, train_index = min_distance_pair
#             test_folder = f"s{self.y_test[test_index]}"
#             train_folder = f"s{self.y_train[train_index]}"

#             test_image_path = os.path.join('dataset', test_folder, f"{test_index % 10 + 1}.pgm")
#             train_image_path = os.path.join('dataset', train_folder, f"{train_index % 10 + 1}.pgm")

#             test_image = Image.open(test_image_path)
#             train_image = Image.open(train_image_path)

#             plt.subplot(1, 2, 1)
#             plt.imshow(test_image, cmap='gray')
#             plt.title(f'Test Image (Actual: {test_folder}, Predicted: {self.y_train[train_index]})')
#             plt.axis('off')

#             plt.subplot(1, 2, 2)
#             plt.imshow(train_image, cmap='gray')
#             plt.title(f'Predicted Train Image (Folder: {train_folder})')
#             plt.axis('off')

#             plt.tight_layout()
#             plt.show()
#         else:
#             print(f'No true positive matches found for {distance}.')


# fdm = FaceRecognition()
# fdm.feature_extraction()
# #fdm.model_selection(model='SVM', param={'C': 1, 'kernel': 'linear'}, verbose=True)
# fdm.model_selection(model='Distance Based')

# class FaceRecognition:
#     def get_train_test_splits(self, folder, train_size=0.8, random_state=42):
#         image_folders = [f for f in os.listdir(folder) if f != 'README']

#         X, y = [], []

#         for f in image_folders:
#             loc = folder + '/' + f
#             folnum = int(f[1:])
#             for file in os.listdir(loc):
#                 file_loc = loc + '/' + file
#                 image = Image.open(file_loc)
#                 pixels = np.asarray(image).reshape(1, -1)

#                 X.append(pixels)
#                 y.append(folnum)

#         X = np.vstack(X)
#         X_train, X_test, y_train, y_test = train_test_split(
#             X, y, train_size=train_size, random_state=random_state)
        
#         y_test = 

#         print('Train and Test splits created', X_train.shape, X_test.shape)
#         return X_train, X_test, y_train, y_test

#     def __init__(self, folder=r'dataset', train_size=0.8, random_state=42):
#         X_train, X_test, y_train, y_test = self.get_train_test_splits(
#             folder=folder, train_size=train_size, random_state=random_state)
#         self.X_train = X_train
#         self.X_test = X_test
#         self.y_train = y_train
#         self.y_test = y_test
#         self.covariance_matrix = None
#         self.eigenval = None
#         self.eigenvec = None
#         self.z = None

#     def feature_extraction(self, n_components=150, verbose=False, pca_method='pca',tsne_params={'n_components': 2, 'perplexity': 30},**kwargs):
#         if verbose:
#             print('Before feature extraction: ',
#                   self.X_train.shape, self.X_test.shape)

#         if pca_method == 'pca':
#             pca = PCA(n_components=n_components)

#         self.X_train = pca.fit_transform(self.X_train)
#         self.X_test = pca.transform(self.X_test)
#         self.covariance_matrix = np.cov(self.X_train)
#         self.eigenval, self.eigenvec = np.linalg.eigh(self.covariance_matrix)
#         explained_variances = pca.explained_variance_ratio_
#         feature_weights = np.sum(
#             (pca.components_ ** 2) * explained_variances.reshape(-1, 1), axis=0)
#         self.z = feature_weights
#         if verbose:
#             print('Total variance explained: ', np.sum(
#                 pca.explained_variance_ratio_))

#     def model_selection(self, model='Random Forest Optuna', param=None, verbose=True, n_trials=100, distance='euclidean', version='normal', plot=False, method='both'):
#         with timer(model):
#             model_name, model_func = self.model_mapping(model)
#             if 'Optuna' in model_name:
#                 param = self.optimize_and_train(model_func, n_trials=n_trials)
#                 print('Best parameters:', param)
#                 if plot:
#                     self.plot_test_image_with_prediction(model = model)
#                 gc.collect()

#             if model_name == 'Distance Based':
#                 acc, report, distances, pairs, y_pred = self.distance_based(
#                     distance=distance, verbose=verbose, version=version)
#                 if plot:
#                     self.plot_distance_based(
#                         distance=distance, distances=distances, pairs=pairs, y_pred=y_pred, method=method)
#             else:
#                 acc, report = self.train_and_report(model_func, param, verbose)

#             print(f'Accuracy of {model_name}: {acc}')
#             if verbose:
#                 print('Classification Report:\n', report)
#             gc.collect()

#         return acc, report

#     def model_mapping(self, model):
#         mapping = {
#             'Random Forest': RandomForestClassifier,
#             'SVM': SVC,
#             'KNN': KNeighborsClassifier,
#             'Naive Bayes': GaussianNB,
#             'Decision Tree': DecisionTreeClassifier,
#             'MLP': MLPClassifier,
#             'Random Forest Optuna': RandomForestClassifier,
#             'SVM Optuna': SVC,
#             'KNN Optuna': KNeighborsClassifier,
#             'Naive Bayes Optuna': GaussianNB,
#             'Decision Tree Optuna': DecisionTreeClassifier,
#             'MLP Optuna': MLPClassifier,
#             'Distance Based': None,
#         }
#         return model, mapping[model]

#     def train_and_report(self, model, param, verbose=False):
#         clf = model(**param)
#         clf.fit(self.X_train, self.y_train)
#         y_pred = clf.predict(self.X_test)
#         accuracy = accuracy_score(self.y_test, y_pred)
#         report = classification_report(
#             self.y_test, y_pred, zero_division=0) if verbose else None
#         return accuracy, report

#     def objective(self, trial, param_search_space, model):
#         param = {key: self.suggest_parameter(
#             trial, key, value[0], value[1]) for key, value in param_search_space.items()}
#         clf = model(**param)
#         return cross_val_score(clf, self.X_train, self.y_train, n_jobs=-1, cv=3).mean()

#     def distance_based(self, distance='euclidean', verbose=False, version='normal'):
#         methods = {
#             'euclidean': self.euclidean_distance,
#             'angle': lambda: self.angle(version=version, verbose=verbose),
#         }

#         if distance in methods:
#             return methods[distance]()
#         else:
#             raise ValueError(f"Unsupported distance metric: {distance}")

#     def angle(self, version='normal', verbose=False):
#         if version == 'normal':
#             return self.cosine_similarity(verbose=verbose)
#         elif version == 'weighted':
#             return self.angle_base(verbose=verbose)

#     def euclidean_distance(self, verbose=False):
#         y_pred, pairs, distances = [], [], []
#         for i in range(self.X_test.shape[0]):
#             dist = [np.linalg.norm(self.X_test[i] - self.X_train[j])
#                     for j in range(self.X_train.shape[0])]
#             y_pred.append(self.y_train[np.argmin(dist)])
#             pairs.append((i, np.argmin(dist)))
#             distances.append(np.min(dist))
#         return self._get_results(y_pred, pairs, distances, verbose)

#     def _get_results(self, y_pred, pairs, distances, verbose):
#         if verbose:
#             return accuracy_score(self.y_test, y_pred), classification_report(self.y_test, y_pred, zero_division=0), distances, pairs, y_pred
#         return accuracy_score(self.y_test, y_pred), None, distances, pairs, y_pred

#     def plot_distance_based(self, distance='euclidean', distances=None, pairs=None, y_pred=None, method='both'):
#         if method == 'both':
#             self.visualize_wrong_and_correct_predictions(
#                 distance, distances, pairs, y_pred)
#         elif method == 'wrong':
#             self._visualize_predictions(
#                 [(i, pairs[i][1]) for i in range(len(pairs)) if self.y_test[i] != y_pred[i]], 'wrong', f'Wrong Predictions - {distance}')
#         elif method == 'correct':
#             self.visualize_correct_top_predictions(
#                 distance, distances, pairs, y_pred)
#         else:
#             print('Invalid method')

#     def visualize_wrong_and_correct_predictions(self, distance, distances, pairs, y_pred):
#         wrong_predictions = [(i, pairs[i][1]) for i in range(
#             len(pairs)) if self.y_test[i] != y_pred[i]]
#         true_predictions = [(i, pairs[i][1]) for i in range(
#             len(pairs)) if self.y_test[i] == y_pred[i]]

#         self._visualize_predictions(
#             wrong_predictions, 'wrong', f'Wrong Predictions - {distance}')
#         self._visualize_predictions(
#             true_predictions, 'correct', f'Correct Predictions - {distance}')

#     def _visualize_predictions(self, predictions, folder_name, title):
#         if len(predictions) > 0:
#             plt.figure(figsize=(15,  5 * len(predictions)))

#             for i, (test_index, train_index) in enumerate(predictions, start=1):
#                 test_folder = f"s{self.y_test[test_index]}"
#                 train_folder = f"s{self.y_train[train_index]}"

#                 test_image_path = os.path.join(
#                     'dataset', test_folder, f"{test_index % 10 + 1}.pgm")
#                 train_image_path = os.path.join(
#                     'dataset', train_folder, f"{train_index % 10 + 1}.pgm")

#                 test_image = Image.open(test_image_path)
#                 train_image = Image.open(train_image_path)

#                 plt.subplot(len(predictions), 2, i * 2 - 1)
#                 plt.imshow(test_image, cmap='gray')
#                 plt.title(
#                     f'Test Image (Actual: {test_folder}, Predicted: {self.y_train[train_index]})')
#                 plt.axis('off')

#                 plt.subplot(len(predictions), 2, i * 2)
#                 plt.imshow(train_image, cmap='gray')
#                 plt.title(f'Predicted Train Image (Folder: {train_folder})')
#                 plt.axis('off')

#             plt.tight_layout()
#             plt.savefig(f'{title}.png')
#             # plt.show()
#         else:
#             print(f'No {folder_name} predictions found.')

#     def visualize_correct_top_predictions(self, distance, distances, pairs, y_pred):
#         true_predictions = [(i, pairs[i][1]) for i in range(
#             len(pairs)) if self.y_test[i] == y_pred[i]]

#         if len(true_predictions) > 0:
#             min_distance_index = min(
#                 true_predictions, key=lambda x: distances[x[0]])[0]
#             min_distance_pair = pairs[min_distance_index]

#             print(
#                 f'Top matching pair for {distance} (Lowest Distance: {distances[min_distance_index]}):')

#             self._visualize_single_pair(min_distance_pair, distance)
#         else:
#             print(f'No true positive matches found for {distance}.')

#     def _visualize_single_pair(self, pair, distance):
#         plt.figure(figsize=(10, 5))

#         test_index, train_index = pair
#         test_folder = f"s{self.y_test[test_index]}"
#         train_folder = f"s{self.y_train[train_index]}"

#         test_image_path = os.path.join(
#             'dataset', test_folder, f"{test_index % 10 + 1}.pgm")
#         train_image_path = os.path.join(
#             'dataset', train_folder, f"{train_index % 10 + 1}.pgm")

#         test_image = Image.open(test_image_path)
#         train_image = Image.open(train_image_path)

#         plt.subplot(1, 2, 1)
#         plt.imshow(test_image, cmap='gray')
#         plt.title(
#             f'Test Image (Actual: {test_folder}, Predicted: {self.y_train[train_index]})')
#         plt.axis('off')

#         plt.subplot(1, 2, 2)
#         plt.imshow(train_image, cmap='gray')
#         plt.title(f'Predicted Train Image (Folder: {train_folder})')
#         plt.axis('off')

#         plt.tight_layout()
#         plt.savefig(f'top_matching_pair_{distance}.png')
#         # plt.show()

from PIL import Image
import optuna
import numpy as np
from numpy import asarray
import os
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from scipy.linalg import eigh
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from scipy.spatial import distance
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.decomposition import PCA as RandomizedPCA
from sklearn.decomposition import SparsePCA, IncrementalPCA, KernelPCA
from sklearn.manifold import TSNE
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import NMF,FastICA
import time
from contextlib import contextmanager
import gc


@contextmanager
def timer(title):
    '''
    This function is used to calculate the time it takes to run a function
    '''

    t0 = time.time()
    yield
    print("{} - done in {:.0f}s".format(title, time.time() - t0))


# def add_data(root):
#     image_folders = os.listdir(root)
#     input_list=[]
#     for i, old_folder_name in enumerate(image_folders, 1):
#         new_folder_name = f"s{i}"
#         new_folder_path = os.path.join(root, new_folder_name)
#         os.rename(os.path.join(root,old_folder_name), new_folder_path)
#         input_list.append(new_folder_path)

#     for input in input_list:
#         for i, filename in enumerate(os.listdir(input), 1):
#             input_path = os.path.join(input, filename)
#             output_path = os.path.splitext(os.path.join(input, f'{i}'))[0] + ".pgm"
#             #print(output_path)
#             im = Image.open(input_path)
#             im = im.resize((92, 112))
#             im = im.convert("L")  # Ensure consistent color mode
#             im.save(output_path)
#             os.remove(input_path)

# add_data(root= r"C:\Users\KyThuat88\Downloads\105_classes_pins_dataset")

class FaceRecognition:
    def get_train_test_splits(self, folder, train_size=0.8, random_state=42):
        image_folders = [f for f in os.listdir(folder) if f != 'README']

        X, y = [], []

        for f in image_folders:
            loc = folder + '/' + f
            folnum = int(f[1:])
            for file in os.listdir(loc):
                file_loc = loc + '/' + file
                image = Image.open(file_loc)
                pixels = np.asarray(image).reshape(1, -1)

                X.append(pixels)
                y.append(folnum)

        X = np.vstack(X)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, train_size=train_size, random_state=random_state)

        print('Train and Test splits created', X_train.shape, X_test.shape)
        return X_train, X_test, y_train, y_test

    def __init__(self, folder=r'dataset', train_size=0.8, random_state=42):
        X_train, X_test, y_train, y_test = self.get_train_test_splits(
            folder=folder, train_size=train_size, random_state=random_state)
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.covariance_matrix = None
        self.eigenval = None
        self.eigenvec = None
        self.z = None

    def twodpca(self, n_components=150, verbose=False):
        if verbose:
            print('Before 2DPCA: ', self.X_train.shape, self.X_test.shape)
            self.plot_cumulative_variance(PCA().fit(self.X_train))

        # Step 1: Compute the mean vector
        mean_vector = np.mean(self.X_train, axis=0)

        # Step 2: Compute the scatter matrices
        S_total = np.zeros_like(np.cov(self.X_train.T))
        S_within = np.zeros_like(S_total)
        S_between = np.zeros_like(S_total)

        unique_classes = np.unique(self.y_train)
        for cls in unique_classes:
            class_indices = np.where(np.array(self.y_train) == cls)[0]
            class_images = self.X_train[class_indices]

            # Compute class-specific covariance matrix
            class_covariance = np.cov(class_images.T)

            # Compute scatter matrices
            S_total += class_covariance
            S_within += (class_images -
                         mean_vector).T @ (class_images - mean_vector)
            S_between += len(class_indices) * \
                np.outer(mean_vector, mean_vector)

        # Step 3: Solve the generalized eigenvalue problem
        eigenvalues, eigenvectors = np.linalg.eig(
            np.linalg.inv(S_within) @ S_between)

        # Step 4: Project the data onto the subspace
        self.X_train = np.dot(self.X_train, eigenvectors)
        self.X_test = np.dot(self.X_test, eigenvectors)

        if verbose:
            print('Total variance explained: ', np.sum(
                eigenvalues) / np.trace(S_total))
            self.plot_cumulative_variance(PCA().fit(self.X_train))
    
    def modular_pca(self,n_components = 150,verbose = False):
        sub_train = np.split(self.X_train,4,axis = 1)
        sub_test = np.split(self.X_test,4,axis = 1)
        new = []
        new_test = []
        for i in range(len(sub_train)):
            pca = PCA(n_components= n_components)
            pca.fit(sub_train[i])
            sub_train[i] = pca.transform(sub_train[i])
            sub_test[i] = pca.transform(sub_test[i])
            new.append(sub_train[i])
            new_test.append(sub_test[i])
        self.X_train = np.concatenate(new,axis = 1)
        self.X_test = np.concatenate(new_test,axis = 1)
        self.covariance_matrix = np.cov(self.X_train)
        self.eigenval, self.eigenvec = np.linalg.eigh(self.covariance_matrix)
        explained_variances = pca.explained_variance_ratio_
        feature_weights = np.sum(
            (pca.components_ ** 2) * explained_variances.reshape(-1, 1), axis=0)
        self.z = feature_weights

        if verbose:
            print('Total variance explained: ', np.sum(
                pca.explained_variance_ratio_))
            self.plot_cumulative_variance(PCA().fit(self.X_train))

    def randomized_pca(self,n_components = 150, verbose = False): 
        if verbose:
            print('Before Randomized PCA: ', self.X_train.shape, self.X_test.shape)
        pca = RandomizedPCA(n_components=n_components)
        self.X_train = pca.fit_transform(self.X_train)
        self.X_test = pca.transform(self.X_test)
        self.covariance_matrix = np.cov(self.X_train)
        self.eigenval, self.eigenvec = np.linalg.eigh(self.covariance_matrix)
        explained_variances = pca.explained_variance_ratio_
        feature_weights = np.sum(
            (pca.components_ ** 2) * explained_variances.reshape(-1, 1), axis=0)
        self.z = feature_weights
        if verbose:
            print('Total variance explained: ', np.sum(
                pca.explained_variance_ratio_))
    def sparse_pca(self, n_components = 50,alpha=1, verbose = False):
        if verbose:
            print('Before Sparse PCA: ', self.X_train.shape, self.X_test.shape)
        pca = SparsePCA(n_components=n_components,alpha=alpha)
        self.X_train = pca.fit(self.X_train)
        components = pca.components_
        covariance_matrix = np.cov(components)
        eigenvalues, _ = np.linalg.eig(covariance_matrix)
        eigenvalues_sorted = np.sort(eigenvalues)[::-1]
        explained_variance_ratio = eigenvalues_sorted / np.sum(eigenvalues_sorted)
        self.X_train= pca.transform(self.X_train)
        self.X_test = pca.transform(self.X_test)
        if verbose:
            print('Total variance explained: ', np.sum(explained_variance_ratio))
        
    # def tune_sparse_pca(self, param_grid, verbose=False):
    #     pca = SparsePCA()
    #     grid_search = GridSearchCV(pca, param_grid, cv=3, scoring='explained_variance', verbose=verbose, n_jobs=-1)
    #     grid_search.fit(self.X_train)

    #     # Get the best parameters from the grid search
    #     best_params = grid_search.best_params_

    #     if verbose:
    #         print('Best Parameters:', best_params)

    #     # Apply sparse PCA with the best parameters
    #     self.sparse_pca(n_components=best_params['n_components'], alpha=best_params['alpha'], verbose=verbose)
            
    def kernel_pca(self, n_components = 150,kernel='linear', verbose=False):
        if verbose:
            print('Before Kernel PCA: ', self.X_train.shape, self.X_test.shape)
        pca = KernelPCA(n_components=n_components,kernel=kernel)
        pca.fit(self.X_train)
        self.eigenval=pca.eigenvalues_
        self.X_train = pca.transform(self.X_train)
        self.X_test = pca.transform(self.X_test)
        explained_variance_ratio=self.eigenval/np.sum(self.eigenval)

        if verbose:
            print('Total variance explained: ', np.sum(explained_variance_ratio))
        
    def incremental_pca(self, n_components = 150,batch_size=None, verbose = False):
        if verbose:
            print('Before Incremental PCA: ', self.X_train.shape, self.X_test.shape)
        pca = IncrementalPCA(n_components=n_components)
        self.X_train = pca.fit_transform(self.X_train)
        self.X_test = pca.transform(self.X_test)
        self.covariance_matrix = np.cov(self.X_train)
        self.eigenval, self.eigenvec = np.linalg.eigh(self.covariance_matrix)
        explained_variances = pca.explained_variance_ratio_
        feature_weights = np.sum(
            (pca.components_ ** 2) * explained_variances.reshape(-1, 1), axis=0)
        self.z = feature_weights
        if verbose:
            print('Total variance explained: ', np.sum(
                pca.explained_variance_ratio_))
    
    def tsne(self, n_components=2, perplexity=30, verbose=False):
        if verbose:
            print('Before t-SNE: ', self.X_train.shape, self.X_test.shape)
        
        tsne = TSNE(n_components=n_components, perplexity=perplexity)
        combined_data = np.vstack((self.X_train, self.X_test))
        combined_data = tsne.fit_transform(combined_data)
        
        self.X_train = combined_data[:len(self.X_train)]
        self.X_test = combined_data[len(self.X_train):]
        
        if verbose:
            print('t-SNE completed')
    
    def truncated_svd(self, n_components=150, verbose=False):
        if verbose:
            print('Before Truncated SVD: ', self.X_train.shape, self.X_test.shape)
        
        svd = TruncatedSVD(n_components=n_components)
        self.X_train = svd.fit_transform(self.X_train)
        self.X_test = svd.transform(self.X_test)
        
        if verbose:
            print('Truncated SVD completed')

    def nmf(self,n_components = 150, verbose = False):
        if verbose:
            print('Before NMF:', self.X_train.shape, self.X_test.shape)
        nmf = NMF(n_components=n_components)
        self.X_train = nmf.fit_transform(self.X_train)
        self.X_test = nmf.transform(self.X_test)

    def ica(self,n_components = 150,verbose = False):
        ica = FastICA(n_components=n_components)
        self.X_train = ica.fit_transform(self.X_train)
        self.X_test = ica.transform(self.X_test)


    def feature_extraction(self, n_components=150, verbose=False, pca_method='pca',tsne_params={'n_components': 2, 'perplexity': 30},**kwargs):
        if verbose:
            print('Before feature extraction: ',
                  self.X_train.shape, self.X_test.shape)

        if pca_method == 'pca':
            pca = PCA(n_components=n_components)
        elif pca_method == 'twodpca':
            self.twodpca(n_components=n_components, verbose=verbose)
            return
        elif pca_method == 'modular':
            self.modular_pca(n_components=150, verbose=False)
            return
        elif pca_method == 'randomized':
            self.randomized_pca(n_components=n_components,verbose = verbose,**kwargs)
            return
        elif pca_method == 'sparse':
            #self.sparse_pca(n_components=n_components,verbose=verbose,**kwargs)
            param_grid = {
                'n_components': [50, 100, 150],
                'alpha': [0.1, 1, 10]}
            self.tune_sparse_pca(param_grid=param_grid,verbose=verbose)
            return
        elif pca_method == 'kernel':
            self.kernel_pca(n_components=n_components,verbose=verbose,**kwargs)
            return
        elif pca_method == 'incremental':
            self.incremental_pca(n_components=n_components,verbose=verbose,**kwargs)
            return
        elif pca_method == 'tsne':
            self.tsne(**tsne_params)
            return
        elif pca_method == 'svd':
            self.truncated_svd(n_components=n_components,verbose=verbose,**kwargs)
            return
        elif pca_method == 'nmf':
            self.nmf(n_components=n_components,verbose=verbose,**kwargs)
            return
        elif pca_method == 'ica':
            self.ica(n_components=n_components,**kwargs)
            return 

        self.X_train = pca.fit_transform(self.X_train)
        self.X_test = pca.transform(self.X_test)
        self.covariance_matrix = np.cov(self.X_train)
        self.eigenval, self.eigenvec = np.linalg.eigh(self.covariance_matrix)
        explained_variances = pca.explained_variance_ratio_
        feature_weights = np.sum(
            (pca.components_ ** 2) * explained_variances.reshape(-1, 1), axis=0)
        self.z = feature_weights
        if verbose:
            print('Total variance explained: ', np.sum(
                pca.explained_variance_ratio_))

    def plot_cumulative_variance(self, pca):
        plt.figure(figsize=(10, 7))
        plt.ylim(0.0, 1.1)
        plt.plot(np.cumsum(pca.explained_variance_ratio_), linewidth=3)
        plt.xlabel('number of components', fontsize=21)
        plt.ylabel('cumulative explained variance', fontsize=21)
        plt.title('Scree Plot using PCA', fontsize=24)
        plt.rc('font', size=16)
        plt.grid()
        plt.savefig('scree_plot.png')
        plt.show()

    def model_selection(self, model='Random Forest Optuna', param=None, verbose=True, n_trials=100, distance='euclidean', version='normal', plot=False, method='both'):
        with timer(model):
            model_name, model_func = self.model_mapping(model)
            if 'Optuna' in model_name:
                param = self.optimize_and_train(model_func, n_trials=n_trials)
                print('Best parameters:', param)
                if plot:
                    _, _, _, predicted_class = self.train_and_report(model_func, param, verbose=False)
                    self.plot_misclassified_images_ml(y_predict=predicted_class)
                gc.collect()

            if model_name == 'Distance Based':
                acc, report, distances, pairs, y_pred = self.distance_based(
                    distance=distance, verbose=verbose, version=version)
                if plot:
                    self.plot_distance_based(
                        distance=distance, distances=distances, pairs=pairs, y_pred=y_pred, method=method)
            else:
                acc, report, predicted_class, = self.train_and_report(model_func, param, verbose)
                self.plot_misclassified_images_ml(y_predict=predicted_class)

            print(f'Accuracy of {model_name}: {acc}')
            if verbose:
                print('Classification Report:\n', report)
            gc.collect()

        return acc, report

    def model_mapping(self, model):
        mapping = {
            'Random Forest': RandomForestClassifier,
            'SVM': SVC,
            'KNN': KNeighborsClassifier,
            'Naive Bayes': GaussianNB,
            'Decision Tree': DecisionTreeClassifier,
            'MLP': MLPClassifier,
            'Random Forest Optuna': RandomForestClassifier,
            'SVM Optuna': SVC,
            'KNN Optuna': KNeighborsClassifier,
            'Naive Bayes Optuna': GaussianNB,
            'Decision Tree Optuna': DecisionTreeClassifier,
            'MLP Optuna': MLPClassifier,
            'Distance Based': None,
        }
        return model, mapping[model]

    def train_and_report(self, model, param, verbose=False):
        clf = model(**param)
        clf.fit(self.X_train, self.y_train)
        y_pred = clf.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, y_pred)
        report = classification_report(
            self.y_test, y_pred, zero_division=0) if verbose else None
        return accuracy, report, y_pred

    def optimize_and_train(self, model, n_trials=100):
        param_search_space = self.get_param_search_space(model)
        return self.optimize_and_train_inner(model, param_search_space, n_trials)

    def suggest_parameter(self, trial, param_name, param_type, param_range):
        if param_type == 'int':
            return trial.suggest_int(name=param_name, low=param_range[0], high=param_range[1])
        elif param_type == 'float':
            return trial.suggest_float(name=param_name, low=param_range[0], high=param_range[1], log=param_range[2])
        elif param_type == 'categorical':
            return trial.suggest_categorical(name=param_name, choices=param_range)
        else:
            raise ValueError(f"Unsupported parameter type: {param_type}")

    def objective(self, trial, param_search_space, model):
        param = {key: self.suggest_parameter(
            trial, key, value[0], value[1]) for key, value in param_search_space.items()}
        clf = model(**param)
        return cross_val_score(clf, self.X_train, self.y_train, n_jobs=-1, cv=3).mean()

    def optimize_and_train_inner(self, model, param_search_space, n_trials=100):
        study = optuna.create_study(direction='maximize')

        def objective_wrapper(trial):
            return self.objective(trial, param_search_space, model)

        study.optimize(objective_wrapper, n_trials=n_trials)
        print('Number of finished trials:', len(study.trials))
        print('Best trial:', study.best_trial.params)

        return study.best_trial.params

    def get_param_search_space(self, model):
        param_search_spaces = {
            RandomForestClassifier: {
                'n_estimators': ('int', (200, 500)),
                'max_depth': ('int', (10, 50)),
                'min_samples_split': ('int', (2, 10)),
                'min_samples_leaf': ('int', (1, 10)),
                'bootstrap': ('categorical', [True, False]),
            },
            SVC: {
                'C': ('float', (1e-2, 1e2, 'log')),
                'gamma': ('float', (1e-5, 1e1, 'log')),
                'kernel': ('categorical', ['rbf', 'linear', 'poly', 'sigmoid']),
            },
            KNeighborsClassifier: {
                'n_neighbors': ('int', (1, 50)),
            },
            GaussianNB: {
                'var_smoothing': ('float', (1e-5, 1e-1, 'log')),
            },
            DecisionTreeClassifier: {
                'max_depth': ('int', (2, 32)),
                'min_samples_split': ('int', (2, 10)),
                'min_samples_leaf': ('int', (1, 10)),
                'criterion': ('categorical', ['gini', 'entropy']),
            },
            MLPClassifier: {
                'hidden_layer_sizes': ('int', (1, 100)),
                'activation': ('categorical', ['identity', 'logistic', 'tanh', 'relu']),
                'solver': ('categorical', ['lbfgs', 'sgd', 'adam']),
                'alpha': ('float', (1e-5, 1e-1, 'log')),
                'learning_rate': ('categorical', ['constant', 'invscaling', 'adaptive']),
                'max_iter': ('int', (100, 1000)),
            },
        }
        return param_search_spaces.get(model, {})

    def distance_based(self, distance='euclidean', verbose=False, version='normal'):
        methods = {
            'euclidean': self.euclidean_distance,
            'minkowski': self.minkowski_distance,
            'mahala': self.simplified_mahala,
            'angle': lambda: self.angle(version=version, verbose=verbose),
            'sse': lambda: self.sse(version=version, verbose=verbose)
        }

        if distance in methods:
            return methods[distance]()
        else:
            raise ValueError(f"Unsupported distance metric: {distance}")

    def angle(self, version='normal', verbose=False):
        if version == 'normal':
            return self.cosine_similarity(verbose=verbose)
        elif version == 'weighted':
            return self.angle_base(verbose=verbose)

    def euclidean_distance(self, verbose=False):
        y_pred, pairs, distances = [], [], []
        for i in range(self.X_test.shape[0]):
            dist = [np.linalg.norm(self.X_test[i] - self.X_train[j])
                    for j in range(self.X_train.shape[0])]
            y_pred.append(self.y_train[np.argmin(dist)])
            pairs.append((i, np.argmin(dist)))
            distances.append(np.min(dist))
        return self._get_results(y_pred, pairs, distances, verbose)

    def cosine_similarity(self, verbose=False):
        y_pred, pairs, distances = [], [], []
        for i in range(self.X_test.shape[0]):
            sim = [np.dot(self.X_test[i], self.X_train[j]) / (np.linalg.norm(self.X_test[i])
                                                              * np.linalg.norm(self.X_train[j])) for j in range(self.X_train.shape[0])]
            y_pred.append(self.y_train[np.argmax(sim)])
            pairs.append((i, np.argmax(sim)))
            distances.append(np.max(sim))
        return self._get_results(y_pred, pairs, distances, verbose)

    def minkowski_distance(self, verbose=False):
        y_pred, pairs, distances = [], [], []
        for i in range(self.X_test.shape[0]):
            dist = [np.sum(np.abs(self.X_test[i] - self.X_train[j])**3)**(1/3)
                    for j in range(self.X_train.shape[0])]
            y_pred.append(self.y_train[np.argmin(dist)])
            pairs.append((i, np.argmin(dist)))
            distances.append(np.min(dist))
        return self._get_results(y_pred, pairs, distances, verbose)

    def angle_base(self, verbose=False):
        y_pred, pairs, distances = [], [], []
        z = self.z
        for i in range(self.X_test.shape[0]):
            dis = [np.dot(z[i], np.dot(self.X_test[i], self.X_train[j])) / (np.linalg.norm(
                self.X_test[i]) * np.linalg.norm(self.X_train[j])) for j in range(self.X_train.shape[0])]
            y_pred.append(self.y_train[np.argmax(dis)])
            pairs.append((i, np.argmax(dis)))
            distances.append(np.max(dis))
        return self._get_results(y_pred, pairs, distances, verbose)

    def simplified_mahala(self, verbose=False):
        y_pred, pairs, distances = [], [], []
        z_i = self.z
        for i in range(self.X_test.shape[0]):
            distance = [np.dot(z_i[j], np.dot(self.X_test[i], self.X_train[j]))
                        for j in range(self.X_train.shape[0])]
            y_pred.append(self.y_train[np.argmax(distance)])
            pairs.append((i, np.argmax(distance)))
            distances.append(np.max(distance))
        return self._get_results(y_pred, pairs, distances, verbose)

    def sse(self, version, verbose=False):
        y_pred, pairs, distances = [], [], []
        for i in range(self.X_test.shape[0]):
            dist = [np.sum((self.X_test[i] - self.X_train[j])**2) / (np.sum((self.X_test[i])**2) + np.sum((self.X_train[j])**2))
                    if version == 'modified' else np.sum((self.X_test[i] - self.X_train[j])**2) for j in range(self.X_train.shape[0])]
            y_pred.append(self.y_train[np.argmin(dist)])
            pairs.append((i, np.argmin(dist)))
            distances.append(np.min(dist))
        return self._get_results(y_pred, pairs, distances, verbose)

    def _get_results(self, y_pred, pairs, distances, verbose):
        if verbose:
            return accuracy_score(self.y_test, y_pred), classification_report(self.y_test, y_pred, zero_division=0), distances, pairs, y_pred
        return accuracy_score(self.y_test, y_pred), None, distances, pairs, y_pred

    def plot_distance_based(self, distance='euclidean', distances=None, pairs=None, y_pred=None, method='both'):
        if method == 'both':
            self.visualize_wrong_and_correct_predictions(
                distance, distances, pairs, y_pred)
        elif method == 'wrong':
            self._visualize_predictions(
                [(i, pairs[i][1]) for i in range(len(pairs)) if self.y_test[i] != y_pred[i]], 'wrong', f'Wrong Predictions - {distance}')
        elif method == 'correct':
            self.visualize_correct_top_predictions(
                distance, distances, pairs, y_pred)
        else:
            print('Invalid method')

    def visualize_wrong_and_correct_predictions(self, distance, distances, pairs, y_pred):
        wrong_predictions = [(i, pairs[i][1]) for i in range(
            len(pairs)) if self.y_test[i] != y_pred[i]]
        true_predictions = [(i, pairs[i][1]) for i in range(
            len(pairs)) if self.y_test[i] == y_pred[i]]

        self._visualize_predictions(
            wrong_predictions, 'wrong', f'Wrong Predictions - {distance}')
        self._visualize_predictions(
            true_predictions, 'correct', f'Correct Predictions - {distance}')

    def _visualize_predictions(self, predictions, folder_name, title):
        if len(predictions) > 0:
            plt.figure(figsize=(15,  5 * len(predictions)))

            for i, (test_index, train_index) in enumerate(predictions, start=1):
                test_folder = f"s{self.y_test[test_index]}"
                train_folder = f"s{self.y_train[train_index]}"

                test_image_path = os.path.join(
                    'dataset', test_folder, f"{test_index % 10 + 1}.pgm")
                train_image_path = os.path.join(
                    'dataset', train_folder, f"{train_index % 10 + 1}.pgm")

                test_image = Image.open(test_image_path)
                train_image = Image.open(train_image_path)

                plt.subplot(len(predictions), 2, i * 2 - 1)
                plt.imshow(test_image, cmap='gray')
                plt.title(
                    f'Test Image (Actual: {test_folder}, Predicted: {self.y_train[train_index]})')
                plt.axis('off')

                plt.subplot(len(predictions), 2, i * 2)
                plt.imshow(train_image, cmap='gray')
                plt.title(f'Predicted Train Image (Folder: {train_folder})')
                plt.axis('off')

            plt.tight_layout()
            plt.savefig(f'{title}.png')
            # plt.show()
        else:
            print(f'No {folder_name} predictions found.')

    def visualize_correct_top_predictions(self, distance, distances, pairs, y_pred):
        true_predictions = [(i, pairs[i][1]) for i in range(
            len(pairs)) if self.y_test[i] == y_pred[i]]

        if len(true_predictions) > 0:
            min_distance_index = min(
                true_predictions, key=lambda x: distances[x[0]])[0]
            min_distance_pair = pairs[min_distance_index]

            print(
                f'Top matching pair for {distance} (Lowest Distance: {distances[min_distance_index]}):')

            self._visualize_single_pair(min_distance_pair, distance)
        else:
            print(f'No true positive matches found for {distance}.')

    def _visualize_single_pair(self, pair, distance):
        plt.figure(figsize=(10, 5))

        test_index, train_index = pair
        test_folder = f"s{self.y_test[test_index]}"
        train_folder = f"s{self.y_train[train_index]}"

        test_image_path = os.path.join(
            'dataset', test_folder, f"{test_index % 10 + 1}.pgm")
        train_image_path = os.path.join(
            'dataset', train_folder, f"{train_index % 10 + 1}.pgm")

        test_image = Image.open(test_image_path)
        train_image = Image.open(train_image_path)

        plt.subplot(1, 2, 1)
        plt.imshow(test_image, cmap='gray')
        plt.title(
            f'Test Image (Actual: {test_folder}, Predicted: {self.y_train[train_index]})')
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.imshow(train_image, cmap='gray')
        plt.title(f'Predicted Train Image (Folder: {train_folder})')
        plt.axis('off')

        plt.tight_layout()
        plt.savefig(f'top_matching_pair_{distance}.png')
        # plt.show()

    def plot_misclassified_images_ml(self, y_predict):
        misclassified_indices = np.where(self.y_test != y_predict)

        plt.figure(figsize=(15, 5 * len(misclassified_indices[0])))

        for i, idx in enumerate(misclassified_indices[0], start=1):
            test_folder = f"s{self.y_test[idx]}"
            predicted_class = f"s{y_predict[idx]}"

            # Load the misclassified test image
            test_image_path = os.path.join('dataset', test_folder, f"1.pgm")
            test_image = Image.open(test_image_path)

            # Load the first image of the predicted class
            predicted_image_path = os.path.join('dataset', predicted_class, f"1.pgm")
            predicted_image = Image.open(predicted_image_path)

            plt.subplot(len(misclassified_indices[0]), 2, i * 2 - 1)
            plt.imshow(test_image, cmap='gray')
            plt.title(f'Misclassified Test Image (Actual: {test_folder}, Predicted: {predicted_class})')
            plt.axis('off')

            plt.subplot(len(misclassified_indices[0]), 2, i * 2)
            plt.imshow(predicted_image, cmap='gray')
            plt.title(f'First Image of Predicted Class ({predicted_class})')
            plt.axis('off')

        plt.tight_layout()
        plt.show()

def test_n_components(min_components=20, max_components=160, step=20, model='Random Forest', folder=r'dataset', train_size=0.8, param=None, verbose=True, n_trials=100, distance='euclidean', version='normal', plot=False, method='both', random_state=42, pca_method='pca'):
    scores = {
        'Accuracy': [],
        'Precision': [],
        'Recall': [],
        'F1-Score': []
    }

    for i in range(min_components, max_components + step, step):
        with timer('Data Preparation'):
            fdm = FaceRecognition(
                folder=folder, train_size=train_size, random_state=random_state)
            gc.collect()

        with timer(f'Number of Components: {i}'):
            with timer('Feature Extraction'):
                fdm.feature_extraction(
                    n_components=i, verbose=verbose, pca_method=pca_method)
                gc.collect()

            with timer('Model Selection'):
                acc, report = fdm.model_selection(
                    model=model, param=param, verbose=verbose, n_trials=n_trials, distance=distance, version=version, plot=plot, method=method)
                gc.collect()

            # Collect scores for each component
            scores['Accuracy'].append(acc)
            report = report.split('\n\n')[2].split('\n')[
                2].split('      ')[1:-1]
            scores['Precision'].append(float(report[0]))
            scores['Recall'].append(float(report[1]))
            scores['F1-Score'].append(float(report[2]))

    # Create a single plot for all scores
    components_range = range(min_components, max_components + step, step)
    plt.figure(figsize=(10, 7))
    plt.plot(components_range, scores['Accuracy'], label='Accuracy')
    plt.plot(components_range, scores['Precision'], label='Precision')
    plt.plot(components_range, scores['Recall'], label='Recall')
    plt.plot(components_range, scores['F1-Score'], label='F1-Score')
    plt.xlabel('Number of Components')
    plt.ylabel('Score')
    plt.title(f'{model} - Scores vs Number of Components')
    plt.legend()
    plt.grid()
    plt.savefig(f'{model}_scores_vs_components.png')
    plt.show()


# face_recognition = FaceRecognition()
# face_recognition.tune_sparse_pca(param_grid, verbose=True) quá lâu
def main(model, folder=r'.\dataset', train_size=0.8, param=None,
         verbose=True, n_components=150, n_trials=20, distance='euclidean', version='normal', plot=False, method='both', random_state=42, pca_method='pca'):
    '''
    This function is used to run the entire project

    Parameters:
    model (str): Model to be used
    folder (str): Folder containing the dataset
    train_size (int): Number of images to be used for training
    param (dict): Parameters for the model
    verbose (bool): Whether to print the classification report or not
    n_components (int): Number of components to be used for feature extraction
    n_trials (int): Number of trials for Optuna
    distance (str): Distance metric to be used
    version (str): Version of the distance metric to be used
    '''
    with timer('Data Preparation'):
        fdm = FaceRecognition(
            folder=folder, train_size=train_size, random_state=random_state)
        gc.collect()
    with timer('Feature Extraction'):
        fdm.feature_extraction(n_components=n_components,
                               verbose=verbose, pca_method=pca_method)
        gc.collect()
    with timer('Model Selection'):
        fdm.model_selection(model=model, param=param,
                            verbose=verbose, n_trials=n_trials, distance=distance, version=version, plot=plot, method=method)
        gc.collect()


if __name__ == '__main__':
    '''Để test n component 1 loạt các model với param có vẻ ổn'''
    # best_params = {
    #     'SVM': {'C': 68.62612780012446, 'gamma': 0.0028340054896037214, 'kernel': 'linear'},
    #     'KNN': {'n_neighbors': 1},
    #     'MLP': {'hidden_layer_sizes': 22, 'activation': 'relu', 'solver': 'lbfgs', 'alpha': 0.09106984998817184, 'learning_rate': 'adaptive', 'max_iter': 649},
    #     'Naive Bayes': {'var_smoothing': 0.016289768779840178},
    #     'Decision Tree': {'max_depth': 26, 'min_samples_split': 2, 'min_samples_leaf': 1, 'criterion': 'entropy'},
    #     'Random Forest': {'n_estimators': 367, 'max_depth': 28, 'min_samples_split': 2, 'min_samples_leaf': 1, 'bootstrap': True}
    # }
    # for i in list(best_params.keys()):
    #     test_n_components(min_components=20, max_components=150, step=20, model=i, folder=r'.\dataset', train_size=0.8,
    #                       param=best_params[i], verbose=True, n_trials=20, random_state=42, pca_method='pca', plot=False)

    '''nếu không test component và chỉ chạy bình thường để lấy kết quả'''
    # for i in list(best_params.keys()):
    #     main(model=i, folder=r'.\dataset', train_size=0.8,
    #          param=best_params[i], n_components=70, random_state=0, verbose=False)
    #main(model = 'Distance Based',folder=r'.\dataset', train_size=0.8,distance='euclidean', n_components=150, random_state=42, verbose=False, plot=False,method='wrong', pca_method="modular")
    main(model='SVM', folder=r'.\dataset', train_size=0.8, param={'C': 68.62612780012446, 'gamma': 0.0028340054896037214, 'kernel': 'linear'}, pca_method="pca",n_components=150, plot = True)
    # test_n_components(model = 'SVM', param = {'C': 68.62612780012446, 'gamma': 0.0028340054896037214, 'kernel': 'linear'})
    '''Để in ra 1 loạt distance based'''
    # distance_params = {
    #     'euclidean': {'distance': 'euclidean', 'version': 'normal'},
    #     'minkowski': {'distance': 'minkowski', 'version': 'normal'},
    #     'sse': {'distance': 'sse', 'version': 'normal'},
    #     'sse_modified': {'distance': 'sse', 'version': 'modified'},
    #     'angle': {'distance': 'angle', 'version': 'normal'},
    #     'mahala': {'distance': 'mahala', 'version': 'normal'},
    #     'angle_weighted': {'distance': 'angle', 'version': 'weighted'},
    # }
    # for i in list(distance_params.keys()):
    #     main(model='Distance Based', folder=r'.\dataset', train_size=0.8, version=distance_params[i]['version'],
    #          distance=distance_params[i]['distance'], n_components=150, random_state=42, verbose=False, plot=False)

    '''Để tunning 1 loạt các model'''
    # list_model_optuna = ['SVM Optuna', 'KNN Optuna', 'MLP Optuna',
    #                      'Naive Bayes Optuna', 'Decision Tree Optuna', 'Random Forest Optuna']
    # for i in list_model_optuna:
    #     main(model=i, folder=r'.\dataset', train_size=0.8,
    #          n_components=50, random_state=42, verbose=False, plot=False, n_trials=50)
