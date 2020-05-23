# A web-app built from [streamlit](https://streamlit.io) to interact and visualize machine learning models

[![Build Status](https://travis-ci.com/eric-li18/modelweave.svg?branch=master)](https://travis-ci.com/eric-li18/modelweave)


CI/CD pipeline setup with Travis-CI. Running on a Docker container hosted on [ Heroku ](https://modelweave.herokuapp.com/).

<img src="./images/reg.gif" alt="demo" width="818" height="430"/>

_Polynomial Regression_

<img src="./images/voronoi.gif" alt="demo" width="818" height="430"/>


_Voronoi Tessellation_

## Code Contribution
Basic ML Model code was provided by [Madhu G Nadig's repository](https://github.com/madhug-nadig/Machine-Learning-Algorithms-from-Scratch)

## Progress
- [ ] Show other datasets
- [ ] Structure the application in a more readable manner
- [ ] Graph with lines not points
- [ ] Animate the visualization of Voronoi Tesselation for 1-NN
- [x] Add CI/CD and deploy onto cloud provider
### Feature: Linear Regression
- [ ] Change code from manual to library-optimized?
- [ ] Change time to yearly increments
- [ ] Change graph to interactive
- [ ] Add quick description of $R^2$ value and other concepts
- [ ] Date is int instead of date, change to reflect that

## About
This web-app was created using [streamlit](https://streamlit.io), [matplotlib](https://matplotlib.org),[sci-kit learn](https://scikit-learn.org/stable/), [pandas](https://pandas.pydata.org/) and [NumPy](https://numpy.org). It is currently a WIP with plans to add more features soon!

<!---A practice in utilizing linear regression to predict a candidates GPA based on SAT scores

Dataset was taken from [here](https://www.kaggle.com/luddarell/101-simple-linear-regressioncsv)

![Training Data](./images/training.jpg)
![Testing Data](./images/testing.jpg)


_Training vs. Testing data based on the same regression line (green)_

## Background
This dataset is based on the 2400 SAT score which was [changed in 2005](https://www.nytimes.com/2002/06/23/us/new-sat-writing-test-is-planned.html) to include a new writing section graded out of 800 points (hence the 800 point increase from the previous 1600 points), and then [changed once again in March of 2014](https://apps.washingtonpost.com/g/page/local/key-shifts-of-the-sat-redesign/858/), with one of the changes being a return to the 1600-point system that was previously used. The first updated exam was administered in March of 2016.

--->
