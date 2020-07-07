# A web-app built with [streamlit](https://streamlit.io) to visualize and interact with machine learning models

[![Build Status](https://travis-ci.com/eric-li18/modelweave.svg?branch=master)](https://travis-ci.com/eric-li18/modelweave)


CI/CD pipeline setup with Travis-CI. Running on a Docker container hosted on [ Heroku](https://modelweave.herokuapp.com/).

<img src="./images/reg.gif" alt="demo" width="818" height="430"/>

_Polynomial Regression_

<img src="./images/voronoi.gif" alt="demo" width="818" height="430"/>


_Voronoi Tessellation_

## Running the Code
The above code can be run via the Dockerfile in the project's root directory with:
```
docker build -t modelweave ./
docker run -p 8080:8080 -itd --name modelweave modelweave
```
Alternatively, the code can be run via:
```
pip install -r requirements.txt
streamlit run app.py
```

## Code Contribution
Basic ML code was provided by [Madhu G Nadig's repository](https://github.com/madhug-nadig/Machine-Learning-Algorithms-from-Scratch)

## Progress
- [ ] Structure the application in a more readable manner
- [ ] Add error function equations, and weight calculations
- [ ] Animate the visualization of Voronoi Tesselation for 1-NN
- [x] Add CI/CD and deploy onto cloud provider
- [x] Plot in plotly for interactivity

