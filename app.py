# MAIN APPLICATION
import streamlit as st
import LinearRegression
import AStar
import LogisticRegression
import PolynomialRegression
import NaiveBayes
import KMeansClustering
import KNearestNeighbours

if __name__ == "__main__":
    page_option = st.sidebar.selectbox()

    if page_option.lower() == "logistic regression":
        LogisticRegression.main()
    elif page_option.lower() == "k nearest neighbours":
        KNearestNeighbours.main()
    elif page_option.lower() == "naive bayes":
        NaiveBayes.main()
    elif page_option.lower() == "polynomial regression":
        PolynomialRegression.main()
    elif page_option.lower() == "a-star":
        AStar.main()
    else:
        # Home Page
        LinearRegression.main()