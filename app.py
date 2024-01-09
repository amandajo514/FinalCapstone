import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import ConfusionMatrixDisplay
import pickle
import base64
import matplotlib.pyplot as plt

# Set page title and icon
st.set_page_config(page_title = "Employee Attrition Predictor", page_icon = ":bar_chart:")

#supressing warning that comes when confusion matrix is shown
st.set_option('deprecation.showPyplotGlobalUse', False)

#Sidebar navigation
page = st.sidebar.selectbox("Select a Page", ["🏠 Home", "📂 Data Overview", "📈 EDA", "⚙️ Modeling", "🔮 Make Predictions!"])

#read in the data
df = pd.read_csv('data/final_attrition_df.csv')

# setting color theme
custom_theme = f"""
    <style>
        :root {{
            --primaryColor: #9C8777;
            --backgroundColor: #FFE2D3;
            --secondaryBackgroundColor: #E2C3B2;
            --textColor: #543022;
            --font: sans-serif;
        }}
    </style>
"""

st.markdown(custom_theme, unsafe_allow_html=True)

#build homepage
if page == "🏠 Home":
    st.title("💼 Employee Attrition Predictor")
    # Audio File 9 to 5
    st.write("🎶: 9 to 5 by Dolly Parton and Kelly Clarkson")
    audio_file_path = "audio/9to5dollypar.mp3" 
    st.audio(audio_file_path, format='audio/mp3', start_time=0)
    st.subheader("This app is designed to effectively review the analytics and make predictions on employee attrition throughout the company.")
    st.write("Please use the toggle bar on the left hand side of the page to navigate between the dataset, the analytics on current employees attrition, and making predictions on future employees attrition.")
    
    # Centered image
    st.image("https://www.fintechfutures.com/files/2018/07/Busy-People-FOT-280x198.jpg", 
         caption="Image Caption",
         use_column_width=True,
         output_format="auto",
         width=0.5) 

#build data overview page
if page == "📂 Data Overview":
    st.title("📂 Data Overview")
    st.subheader("About the Data")
    st.write("The dataset at hand encompasses a comprehensive set of attributes related to employees within our organization. These attributes include, but are not limited to, marital status, job title, compensation, education, and more. Each row represents a unique employee, and the primary objective of this dataset is to facilitate the prediction of employee attrition.")
    st.image("data:image/jpeg;base64,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")
    st.link_button("Click here to learn more", "https://www.kaggle.com/datasets/patelprashant/employee-attrition", help = "Employee Attrition Dataset Kaggle Page", type = 'primary')
    st.subheader("Quick Glance at the Data")
    # Display dataset
    if st.checkbox("DataFrame"):
        st.dataframe(df)
    # Column list
    if st.checkbox("Column List"):
        st.code(f"Columns: {df.columns.tolist()}")
        if st.toggle('Further breakdown of columns'):
            num_cols = df.select_dtypes(include = 'number').columns.tolist()
            obj_cols = df.select_dtypes(include = 'object').columns.tolist()
            st.code(f"Numerical Columns: {num_cols} \nObject Columns: {obj_cols}")
    # Shape
    if st.checkbox("Shape"):
        # st.write(f"The shape is {df.shape}") -- could write it out like this or do the next instead:
        st.write(f"There are {df.shape[0]} rows and {df.shape[1]} columns.")

    if st.button("Download Data as CSV"):
    # Create a link for downloading the data
        csv = df.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()  # Convert the DataFrame to CSV, encode as base64
        href = f'<a href="data:file/csv;base64,{b64}" download="employee_attrition_data.csv">Download CSV file</a>'
        st.markdown(href, unsafe_allow_html=True)

#build EDA page
if page == "📈 EDA":
    st.title("📈 EDA")
    num_cols = df.select_dtypes(include='number').columns.tolist()
    obj_cols = df.select_dtypes(include='object').columns.tolist()
    eda_type = st.multiselect("What type of EDA are you interested in exploring?", ["Histogram", "Box Plot", "Scatterplot"])

    # Set a custom color palette with browns and tans
    custom_palette = ['#8C564B', '#D2B48C']

    # HISTOGRAM
    if "Histogram" in eda_type:
        st.subheader("Histograms - Visualizing Numerical Distributions")
        h_selected_col = st.selectbox("Select a numerical column for your Histogram:", num_cols, index=None)
        if h_selected_col:
            chart_title = f"Distribution of {' '.join(h_selected_col.split('_')).title()}"
            if st.toggle("Attrition Hue on Histogram:"):
                st.plotly_chart(px.histogram(df, x=h_selected_col, title=chart_title, color='attrition',
                                              barmode='overlay', color_discrete_sequence=custom_palette))
            else:
                st.plotly_chart(px.histogram(df, x=h_selected_col, title=chart_title, color_discrete_sequence=custom_palette))

    # BOXPLOT
    if "Box Plot" in eda_type:
        st.subheader("Boxplots - Visualizing Numerical Distributions")
        b_selected_col = st.selectbox("Select a numerical column for your Boxplot:", num_cols, index=None)
        if b_selected_col:
            chart_title = f"Distribution of {' '.join(b_selected_col.split('_')).title()}"
            if st.toggle("Attrition Hue On Box Plot"):
                st.plotly_chart(px.box(df, x=b_selected_col, y='attrition', title=chart_title, color='attrition',
                                       color_discrete_sequence=custom_palette))
            else:
                st.plotly_chart(px.box(df, x=b_selected_col, title=chart_title, color_discrete_sequence=custom_palette))

    # SCATTERPLOT
    if "Scatterplot" in eda_type:
        st.subheader("Scatterplots - Visualizing Relationships")
        selected_col_x = st.selectbox("Select x-axis variable:", num_cols, index=None)
        selected_col_y = st.selectbox("Select y-axis variable:", num_cols, index=None)
        if selected_col_x and selected_col_y:
            chart_title = f"Relationship of {selected_col_x} vs {selected_col_y}"
            if st.toggle("Attrition Hue On Scatterplot"):
                st.plotly_chart(px.scatter(df, x=selected_col_x, y=selected_col_y, title=chart_title, color='attrition',
                                    color_discrete_sequence=custom_palette))
            else:
                st.plotly_chart(px.scatter(df, x=selected_col_x, y=selected_col_y, title=chart_title,
                                    color_discrete_sequence=custom_palette))
                
# Build Modeling Page
if page == "⚙️ Modeling":
    st.title("⚙️ Modeling")
    st.markdown("On this page, you can see how well different **machine learning** models make predictions on employee attrition:")
    # Set up X and y
    features = ['age', 'dailyrate', 'distancefromhome', 'environmentsatisfaction', 'hourlyrate', 'jobinvolvement', 'jobsatisfaction', 'monthlyincome', 'monthlyrate', 'numcompaniesworked', 'percentsalaryhike', 'performancerating', 'relationshipsatisfaction', 'stockoptionlevel', 'totalworkingyears', 'trainingtimeslastyear', 'worklifebalance', 'yearsatcompany', 'yearsincurrentrole', 'yearssincelastpromotion','yearswithcurrmanager']
    X = df[features]
    y = df['attrition']
    # Train Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    # Model selection
    model_option = st.selectbox("Select a Model:", ['KNN', 'Logistic Regression', 'Random Forest'], index=None)
    if model_option:
        if model_option == "KNN":
            k_value = st.slider("Select the number of neighbors (k):", 1, 29, 5, 2)
            model = KNeighborsClassifier(n_neighbors=k_value)
        elif model_option == "Logistic Regression":
            model = LogisticRegression()
        elif model_option == "Random Forest":
            model = RandomForestClassifier()
        # create a button & fit your model
        if st.button("Let's see the performance!"):
            model.fit(X_train, y_train)
            # Display results
            st.subheader(f"{model} Evaluation")
            st.text(f"Training Accuracy: {round(model.score(X_train, y_train) * 100, 2)}%")
            st.text(f"Testing Accuracy: {round(model.score(X_test, y_test) * 100, 2)}%")
            # Confusion Matrix
            st.subheader("Confusion Matrix")
            ConfusionMatrixDisplay.from_estimator(model, X_test, y_test, cmap='YlOrBr_r')
            CM_fig = plt.gcf()
            st.pyplot(CM_fig)
    if st.button("Download Model"):
        with open("trained_model.pkl", "wb") as model_file:
            pickle.dump(model, model_file)
        st.success("Model downloaded successfully!")

# Predictions Page
if page == "🔮 Make Predictions!":
    st.title("🔮 Make Employee Predictions")
    st.write("This predictive model estimates the likelihood of employee attrition based on various factors. Adjust the sliders to input employee information, and the model will make a prediction.")
    # Create sliders for user to input data
    age = st.slider("Age", min_value=18, max_value=60, value=30, step=1)
    daily_rate = st.slider("Daily Rate", min_value=102, max_value=1499, value=800, step=1)
    distance_from_home = st.slider("Distance From Home", min_value=1, max_value=29, value=15, step=1)
    environment_satisfaction = st.slider("Environmental Satisfaction", min_value=1, max_value=4, value=2, step=1)
    hourly_rate = st.slider("Hourly Rate", min_value=30, max_value=100, value=60, step=1)
    job_involvement = st.slider("Job Involvement", min_value=1, max_value=4, value=2, step=1)
    job_satisfaction = st.slider("Job Satisfaction", min_value=1, max_value=4, value=2, step=1)
    monthly_income = st.slider("Monthly Income", min_value=1009, max_value=20000, value=10000, step=1)
    monthly_rate = st.slider("Monthly Rate", min_value=2094, max_value=27000, value=10000, step=1)
    num_companies_worked = st.slider("Number Of Companies Worked", min_value=0, max_value=9, value=5, step=1)
    percent_salary_hike = st.slider("Percent Salary Hike", min_value=11, max_value=25, value=20, step=1)
    performance_rating = st.slider("Performance Rating", min_value=3, max_value=4, value=4, step=1)
    relationship_satisfaction = st.slider("Relationship Satisfaction", min_value=1, max_value=4, value=2, step=1)
    stock_option_level = st.slider("Stock Option Level", min_value=0, max_value=3, value=1, step=1)
    total_working_years = st.slider("Total Working Years", min_value=0, max_value=40, value=10, step=1)
    training_times_last_year = st.slider("Training Time Last Year", min_value=0, max_value=6, value=3, step=1)
    work_life_balance = st.slider("Work Life Balance", min_value=1, max_value=4, value=2, step=1)
    years_at_company = st.slider("Years At Company", min_value=0, max_value=40, value=10, step=1)
    years_in_current_role = st.slider("Years In Current Role", min_value=0, max_value=18, value=10, step=1)
    years_since_last_promotion = st.slider("Years Since Last Promotion", min_value=0, max_value=15, value=10, step=1)
    years_with_current_manager = st.slider("Years With Current Manager", min_value=0, max_value=17, value=10, step=1)

    # Must be in order that the model was trained on

    user_input_lr = pd.DataFrame({
        'age': [age],
        'dailyrate': [daily_rate],
        'distancefromhome': [distance_from_home],
        'environmentsatisfaction': [environment_satisfaction],
        'hourlyrate': [hourly_rate],
        'jobinvolvement': [job_involvement],
        'jobsatisfaction': [job_satisfaction],
        'monthlyincome': [monthly_income],
        'monthlyrate': [monthly_rate],
        'numcompaniesworked': [num_companies_worked],
        'percentsalaryhike': [percent_salary_hike],
        'performancerating': [performance_rating],
        'relationshipsatisfaction': [relationship_satisfaction],
        'stockoptionlevel': [stock_option_level],
        'totalworkingyears': [total_working_years],
        'trainingtimeslastyear': [training_times_last_year],
        'worklifebalance': [work_life_balance],
        'yearsatcompany': [years_at_company],
        'yearsincurrentrole': [years_in_current_role],
        'yearssincelastpromotion': [years_since_last_promotion],
        'yearswithcurrmanager': [years_with_current_manager]
    })

    features_lr = ['age', 'dailyrate', 'distancefromhome', 'environmentsatisfaction', 'hourlyrate', 'jobinvolvement', 'jobsatisfaction', 'monthlyincome', 'monthlyrate', 'numcompaniesworked', 'percentsalaryhike', 'performancerating', 'relationshipsatisfaction', 'stockoptionlevel', 'totalworkingyears', 'trainingtimeslastyear', 'worklifebalance', 'yearsatcompany', 'yearsincurrentrole', 'yearssincelastpromotion', 'yearswithcurrmanager']

    X_lr = df[features_lr]
    y_lr = df['attrition']

    model_lr = LogisticRegression()

    if st.button("Make a Prediction! (Attrition: 1 = Employee Leaves | Attrition: 0 = Employee Stays)"):
        model_lr.fit(X_lr, y_lr)  # Fit the model here
        prediction_lr = model_lr.predict(user_input_lr)
        st.write(f"{model_lr} predicts the attrition as {prediction_lr[0]}.")
        st.balloons()
        if prediction_lr[0] == 0:
            st.subheader("Attrition 0 = Employee Will Stay!")
        else:
            st.subheader("Attrition 1 = Employee Will Leave!")
    
    prediction_proba_lr = model_lr.predict_proba(user_input_lr)[:, 1]
    st.write(f"The predicted probability of attrition is: {prediction_proba_lr[0]:.2%}")