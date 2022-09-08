#!/usr/bin/env python3

import math
import numpy as np
from sklearn.linear_model import LinearRegression

oarsmen_numbers = np.array([1, 2, 2, 2, 4, 4, 4, 8])
speeds = np.array([2/(6/60 + 30.74/3600), 2/(6/60 + 8.5/3600), 2/(6/60 + 33.26/3600), 2/(5/60 + 59.72/3600), 2/(5/60 + 37.86/3600), 2/(5/60 + 58.96/3600), 2/(5/60 + 32.03/3600), 2/(5/60 + 18.68/3600)])

print(f"speeds: {speeds}")

def normalize(pair, model):
  weight, record = pair
  log_prediction = model.predict(np.array([math.log10(weight)]).reshape(-1, 1))
  prediction = pow(10.0, log_prediction)
  print(f"Record: {record}, prediction: {prediction}")
  return 100 * (record / prediction - 1)

def regress(x, y):
  model = LinearRegression()

  # log-log
  x_log = np.log10(x)
  y_log = np.log10(y)

  model.fit(x_log.reshape(-1, 1), y_log)

  r_sq = model.score(x_log.reshape(-1, 1), y_log)
  print(f"coefficient of determination: {r_sq}")
  print(f"intercept: {model.intercept_}")
  print(f"slope: {model.coef_}")

  this_normalize = lambda pair: normalize(pair, model)

  zipped_array = np.array(list(zip(x, y)))
  # print(f"Zipped: {zipped_array}")

  normalized = np.apply_along_axis(this_normalize, 1, zipped_array)
  print(f"Normalized: {normalized}")

  # pred_x = 1
  # print(f"prediction for {pred_x}: {model.predict(np.array([pred_x]).reshape(-1,1))}")

regress(oarsmen_numbers, speeds)
