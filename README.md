# TrajectoryForecasting
Source code of paper !["Back to square one: probabilistic trajectory forecasting without bells and whistles"](https://arxiv.org/abs/1812.02984)

Run the script.sh to preprocess the Data.
More information about training the model and testing it will be added soon.

## Model Architecture
![Architecture](http://uupload.ir/files/qfyl_model.png)

### Generated Samples for Mnist Dataset
Given the initial segment (Yellow), the model has generated these samples in an auto-regressive manner.

![Row1c0](./Images/mnist/s1/(0).png)
![Row1c1](./Images/mnist/s1/(1).png)
![Row1c2](./Images/mnist/s1/(2).png)
![Row1c3](./Images/mnist/s1/(3).png)
![Row1c4](./Images/mnist/s1/(4).png)
![Row1c5](./Images/mnist/s1/(5).png)


![Row2c0](./Images/mnist/s2/(0).png)
![Row2c1](./Images/mnist/s2/(1).png)
![Row2c2](./Images/mnist/s2/(2).png)
![Row2c3](./Images/mnist/s2/(3).png)
![Row2c4](./Images/mnist/s2/(4).png)
![Row2c5](./Images/mnist/s2/(5).png)

![Row3c0](./Images/mnist/s3/(0).png)
![Row3c1](./Images/mnist/s3/(1).png)
![Row3c2](./Images/mnist/s3/(2).png)
![Row3c3](./Images/mnist/s3/(3).png)
![Row3c4](./Images/mnist/s3/(4).png)
![Row3c5](./Images/mnist/s3/(5).png)

### Generated Samples for Mnist Dataset
Given the initial segment, which is the path of the object in the first two seconds (Yellow), the model has generated these samples (path of the object in the next four seconds) in an auto-regressive manner.

![Row1c0](./Images/sdd/s1/(0).png)
![Row1c1](./Images/sdd/s1/(1).png)
![Row1c2](./Images/sdd/s1/(2).png)
![Row1c3](./Images/sdd/s1/(3).png)

![Row2c0](./Images/sdd/s2/(0).png)
![Row2c1](./Images/sdd/s2/(1).png)
![Row2c2](./Images/sdd/s2/(2).png)
![Row2c3](./Images/sdd/s2/(3).png)

![Row3c0](./Images/sdd/s3/(0).png)
![Row3c1](./Images/sdd/s3/(1).png)
![Row3c2](./Images/sdd/s3/(2).png)
![Row3c3](./Images/sdd/s3/(3).png)
