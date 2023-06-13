
To test build and run in shell (if Docker is installed locally) -
```sh
% docker build -t nom-docker .
% docker run -p 8050:8050 nom-docker
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

