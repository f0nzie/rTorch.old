

## Build Travis Docker image

From the project directory:

```
 docker build -f docker/Dockerfile -t offline-travis-r .
```

`-f docker/Dockerfile`: location of the Dockerfile

`-t offline-travis-r`: tag with the name of the image

`.`: start at current folder





```
if [ "$(docker inspect -f '{{.State.Running}}' offline-travis-kont-r 2>/dev/null)" = "true" ]; then echo Stopping container; docker stop offline-travis-kont-r; else echo Nothing to stop; fi; docker build -f docker/Dockerfile -t offline-travis-r . 
```

## Run the Travis container

```
docker run --rm --name offline-travis-kont-r -dit offline-travis-r /sbin/init
```

`docker run`: run command in **new** container

`--rm`: remove after run

`--name offline-travis-kont-r`: name of the container

`-dit`: detach, interactive terminal

`offline-travis-r`: name of the image



## Execute bash terminal

```
docker exec -it offline-travis-kont-r bash -l
```

`docker exec`: run command in **existing** container

`-it`: interactive terminal

`offline-travis-kont-r`: name of the container

`bash -l`: open bash and login