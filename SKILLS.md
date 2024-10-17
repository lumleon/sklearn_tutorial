# Probability Calibration
Some machine learning model classification task models provide probability output for each class. The problem with the probability estimation output is that it is not necessarily well-calibrated, which means that it does not reflect the actual likelihood of the output.

For example, your model might provide 95% of the “fraud” class output, but only 70% of that prediction is correct. Probability calibration would aim to adjust the probabilities to reflect the actual likelihood.

There are a few calibration methods, although the most common are the sigmoid calibration and the isotonic regression. The following code uses Scikit-Learn to calibrate the technique in the classifier.

  from sklearn.calibration import CalibratedClassifierCV
  from sklearn.svm import SVC
   
  svc = SVC(probability=False)
  calibrated_svc = CalibratedClassifierCV(base_estimator=svc, method='sigmoid', cv=5)
  calibrated_svc.fit(X_train, y_train)
  probabilities = calibrated_svc.predict_proba(X_test)
  You can change the model as long as it provides probability output. The method allows you to switch between the “sigmoid” or “isotonic”.
  
  For example, here is a Random Forest classifier with isotonic calibration.
  
  from sklearn.calibration import CalibratedClassifierCV
  from sklearn.ensemble import RandomForestClassifier
   
  rf = RandomForestClassifier(random_state=42)
  calibrated_rf = CalibratedClassifierCV(base_estimator=rf, method='isotonic', cv=5)
  calibrated_rf.fit(X_train, y_train)
  probabilities = calibrated_rf.predict_proba(X_test)
If your model does not provide the desired prediction, consider calibrating your classifier.
