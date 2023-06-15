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

## Testing in the cloud

In Google Cloud Shell Terminal -
```sh
$ git clone https://github.com/<USER>/<REPO>.git
$ cd <REPO>
$ git checkout <BRANCH>
$ docker build -t <TAG> .
```

Google Cloud Shell was a web-preview feature (VPN) that can be used to run web applications and view in your browser (not visible to other users). See <https://cloud.google.com/shell/docs/using-web-preview>

```sh
$ docker run -p 8080:80 <TAG>
```

## Deploy to Google Cloud Run

There are two ways to deploy a Docker container in Google Cloud Run, either specify a Docker image in a repository, e.g. on Docker Hub, or allow Cloud Run to build the image from a suitable GitHub repository.  

Keep costs down by specifying 0 as the minimum number of instances.  I typically set the maximum to 5 rather than the default of 100.

## Deploy to AWS Lightsail

```
%aws aws lightsail push-container-image --region eu-west-2 --service-name container-demo --image msaunby/nom:latest --label nom 
```

This will add the image to the available images at <https://lightsail.aws.amazon.com/ls/webapp/eu-west-2/container-services/container-demo/images>

To complete deployment go to <https://lightsail.aws.amazon.com/ls/webapp/eu-west-2/container-services/container-demo/deployments>  Note that the port used by this container is 8050.

