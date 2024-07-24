import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

uploaded_file = st.sidebar.file_uploader("Upload Data File", type="csv")

if uploaded_file is not None:

    tab1, tab2 = st.tabs(["Train", "Inference"])
    df = pd.read_csv(uploaded_file)
    model = LinearRegression()

    with tab1:

        col1, col2 = st.columns(2)

        with col1:

            st.write(df)

        with col2:

            fture = df.columns

            features = st.multiselect("Select Features", options=fture[:-1])
            if len(features) != 0:
                X = df.loc[:, features]
                y = df['Sales']

                model.fit(X, y)
                y_pred = model.predict(X)

                st.info(f"Model trained, MAE: {mean_absolute_error(y, y_pred):.3f}, MSE: {mean_squared_error(y, y_pred):.3f}")

                fig = go.Figure()
                if len(features) == 1:
                    fig.add_trace(go.Scatter(x=X[features[0]], y=y, mode='markers', name='Actual'))

                    xx = np.linspace(X[features[0]].min(), X[features[0]].max(), 100).reshape((-1, 1))
                    y_pred_line = model.predict(xx)

                    fig.add_trace(go.Scatter(x=xx.flatten(), y=y_pred_line, mode='lines', name='Predicted', line=dict(color='red')))

                    fig.update_layout(
                        scene=dict(
                            xaxis_title=features[0],
                            yaxis_title='Sales'
                        )
                    )

                elif len(features) == 2:
                    fig.add_trace(go.Scatter3d(x=X[features[0]], y=X[features[1]], z=y, mode='markers', name='Actual'))

                    xx = np.linspace(X[features[0]].min(), X[features[0]].max(), 100)
                    yy = np.linspace(X[features[1]].min(), X[features[1]].max(), 100)
                    xx, yy = np.meshgrid(xx, yy)
                    zz = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

                    fig.add_trace(go.Surface(x=xx, y=yy, z=zz, opacity=0.5, showscale=False, name='Predicted'))

                    fig.update_layout(
                        scene=dict(
                            xaxis_title=features[0],
                            yaxis_title=features[1],
                            zaxis_title='Sales'
                        )
                    )

                st.plotly_chart(fig)




    with tab2:
        X = df[['Radio', 'TV']]
        y = df['Sales']

        model.fit(X, y)

        col1, col2 = st.columns(2)
        with col1:
            radio_input = st.number_input('Radio', min_value=0.0, max_value=300.0, step=0.1)
        with col2:
            tv_input = st.number_input('TV', min_value=0.0, max_value=50.0, step=0.1)

        if tv_input == 0 and radio_input == 0:
            st.error('Please input: Radio or TV')
        else:
            input_data = np.array([[radio_input, tv_input]])
            prediction = model.predict(input_data)
            st.success(f'Prediction: {prediction[0]:.3f}$')

