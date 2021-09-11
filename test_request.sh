#!/usr/bin/env bash

curl -X 'POST' \
  'http://0.0.0.0/predict' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
        "alcohol": 13.2,
        "malic_acid": 1.7,
        "ash": 2.1,
        "alcalinity_of_ash": 11.2,
        "magnesium": 100.0,
        "total_phenols": 2.6,
        "flavanoids": 2.7,
        "nonflavanoid_phenols": 0.2,
        "proanthocyanins": 1.2,
        "color_intensity": 4.3,
        "hue": 1.0,
        "od280_od315_of_diluted_wines": 3.4,
        "proline": 1050.0
      }'