## Online Prediction Deployment: FastAPI and Docker

Deployment of a webserver that hosts a predictive model trained on the wine dataset using FastAPI and Docker.


## Build the image

With the `Dockerfile` ready, it is time to build the image.

```bash
docker build -t <NAME>:<TAG> .
```


## Run the container

Now that the image has been successfully built it is time to run a container out of it. You can do so by using the following command:

```bash
docker run --rm -p 80:80  <NAME>:<TAG>
```

Notes:
- `--rm`: Delete this container after stopping running it. This is to avoid having to manually delete the container. Deleting unused containers helps your system to stay clean and tidy.
- `-p 80:80`: This flags performs an operation knows as port mapping. The container, as well as your local machine, has its own set of ports. So you are able to access the port 80 within the container, you need to map it to a port on your computer. In this case it is mapped to the port 80 in your machine. 

At the end of the command is the name and tag of the image you want to run. 

After some seconds the container will start and spin up the server within. You should be able to see FastAPI's logs being printed in the terminal. 

Now head over to [localhost:80](http://localhost:80) and you should see a message about the server spinning up correctly.

In [localhost:80/docs] we can see a user-friendly interface to make requests.


## Make requests to the server

Now that the server is listening to requests on port 80, you can send `POST` requests to it for predicting classes of wine.

Every request should contain the data that represents a wine in `JSON` format like this:

```json
{
  "alcohol":12.6,
  "malic_acid":1.34,
  "ash":1.9,
  "alcalinity_of_ash":18.5,
  "magnesium":88.0,
  "total_phenols":1.45,
  "flavanoids":1.36,
  "nonflavanoid_phenols":0.29,
  "proanthocyanins":1.35,
  "color_intensity":2.45,
  "hue":1.04,
  "od280_od315_of_diluted_wines":2.77,
  "proline":562.0
}
```

This example represents a class 1 wine.

Remember from Course 1 that FastAPI has a built-in client for you to interact with the server. You can use it by visiting [localhost:80/docs](http://localhost:80/docs)

You can also use `curl` and send the data directly with the request like this:

```bash
curl -X 'POST' http://localhost/predict \
  -H 'Content-Type: application/json' \
  -d '{
  "alcohol":12.6,
  "malic_acid":1.34,
  "ash":1.9,
  "alcalinity_of_ash":18.5,
  "magnesium":88.0,
  "total_phenols":1.45,
  "flavanoids":1.36,
  "nonflavanoid_phenols":0.29,
  "proanthocyanins":1.35,
  "color_intensity":2.45,
  "hue":1.04,
  "od280_od315_of_diluted_wines":2.77,
  "proline":562.0
}'
```

Or you can use a `JSON` file to avoid typing a long command like this:

```bash
curl -X POST http://localhost:80/predict \
    -d @./wine-examples/1.json \
    -H "Content-Type: application/json"
```

Let's understand the flags used:
- `-X`: Allows you to specify the request type. In this case it is a `POST` request.
- `-d`: Stands for `data` and allows you to attach data to the request.
- `-H`: Stands for `Headers` and it allows you to pass additional information through the request. In this case it is used to the tell the server that the data is sent in a `JSON` format.