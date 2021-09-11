#!/usr/bin/env bash

IMAGE_NAME='gabriel-luz-wine-model:single-prediction'
CONTAINER_NAME='my_wine_api'

docker build -t $IMAGE_NAME .
docker run --rm --name $CONTAINER_NAME -p 80:80 $IMAGE_NAME