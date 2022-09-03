#!/usr/bin/env python3

import math
import numpy as np
from sklearn.linear_model import LinearRegression

men_weights_snatch = np.array([ 61,  67,  73,  81,  96, 109])
men_records_snatch = np.array([145, 155, 169, 175, 187, 200])

men_weights_clean = np.array([ 55,  61,  67,  73,  81,  89,  96, 109])
men_records_clean = np.array([166, 174, 188, 198, 208, 217, 231, 241])

men_weights_total = np.array([ 55,  61,  67,  73,  81,  89,  96, 109])
men_records_total = np.array([294, 318, 339, 364, 378, 392, 416, 435])

women_weights_snatch = np.array([ 49,  55,  59,  64,  76])
women_records_snatch = np.array([ 96, 102, 110, 117, 124])

women_weights_clean = np.array([ 49,  55,  59,  64,  71,  76])
women_records_clean = np.array([119, 129, 140, 145, 152, 156])

women_weights_total = np.array([ 49,  55,  59,  64,  71,  76])
women_records_total = np.array([213, 227, 247, 261, 267, 278])

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

regress(men_weights_snatch, men_records_snatch)
regress(men_weights_clean, men_records_clean)
regress(men_weights_total, men_records_total)

regress(women_weights_snatch, women_records_snatch)
regress(women_weights_clean, women_records_clean)
regress(women_weights_total, women_records_total)
