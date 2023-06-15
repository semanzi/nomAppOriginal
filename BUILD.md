# How to build and deploy

The ```Dockerfile``` here should enable simple building and deployment on a range of cloud hosting
services. It can also be used for automated (DevOps) build and deployment, for example with GitHub Actions.


## Running locally
There is no requirement to have a local Docker environment to build and run the container locally,
just run the Python app as usual, e.g.

```sh
% pipenv run python app/main.py
```

### With Docker
If you do wish to run in a container, if Docker is installed locally, use the following.
Note that the container uses port 80, but you will likely want to forward this to port 8080
or similiar.

```sh
% docker build -t nom-docker .
% docker run -p 8080:80 nom-docker
```

In Google Cloud Shell -
```sh

```

To deploy to AWS Lightsail

```
%aws aws lightsail push-container-image --region eu-west-2 --service-name container-demo --image msaunby/nom:latest --label nom 
```

This will add the image to the available images at <https://lightsail.aws.amazon.com/ls/webapp/eu-west-2/container-services/container-demo/images>

To complete deployment go to <https://lightsail.aws.amazon.com/ls/webapp/eu-west-2/container-services/container-demo/deployments>  Note that the port used by this container is 8050.

