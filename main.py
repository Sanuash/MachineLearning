import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from pathlib import Path
import pickle
import re

st.title("Предсказание стоимости автомобиля по параметрам")

st.header("Добро пожаловать!")

def process_torque(value):
    if pd.isna(value):
        return np.nan, np.nan

    text = str(value).lower().replace(',', '')

    numbers = [float(b or a) for a, b in re.findall(r'(\d+(?:\.\d+)?)(?:-(\d+(?:\.\d+)?))?', text)]

    # Крутящий момент всегда меньше количества оборотов в минуту
    moment = min(numbers)
    rpm = max(numbers)

    is_kgm = 'kgm' in text

    if is_kgm and not pd.isna(moment):
        moment *= 9.81

    return moment, rpm

def drop_measurement(df, col):
  return df[col].astype("str").apply(lambda x: x.split()[0]).astype(float)

def data_preprocessing(df):
    for i in range(len(df)):
        try:
            ffil_value = float(df.loc[i, 'max_power'][:-4])
            df.loc[i, 'max_power'] = ffil_value

        except:
            if pd.isna(df.loc[i, 'max_power']) or df.loc[i, 'max_power'] == 'nan':
                continue
            elif df.loc[i, 'max_power'] == '0':
                ffil_value = float(df.loc[i, 'max_power'])
                df.loc[i, 'max_power'] = ffil_value
            else:
                df.loc[i, 'max_power'] = 0

    df = df.drop_duplicates(subset=df.columns.drop('selling_price')).reset_index(drop=True)

    for col in ["mileage", "engine", "max_power"]:
        df[col] = drop_measurement(df, col)

    df[['torque', 'max_torque_rpm']] = df['torque'].apply(lambda x: pd.Series(process_torque(x)))

    df.fillna(df[["mileage", "engine", "max_power", "torque", "seats", "max_torque_rpm"]].median(), inplace=True)

    df.seats = df.seats.astype(int)

    df.engine = df.engine.astype(int)

    return df

def business_metrics(y_true, y_pred):

   predictions = abs(y_pred - y_true) / y_true

   return sum(predictions <= 0.1) / len(predictions)

def asymmetric_exp_error(y_true, y_pred, alpha=1.5):
    r = y_true - y_pred
    rel_error = np.abs(r) / y_true
    weights = np.where(r > 0, alpha, 1.0)
    return 1 - np.mean(weights * rel_error)

# Готовим данные для предсказаний
def proces_cat_features(df_train, df_test):
    X_train_cat = df_train.drop(columns=["selling_price"])
    X_test_cat = df_test.drop(columns=["selling_price"])

    X_train_cat["seats"] = X_train_cat["seats"].astype(str)
    X_test_cat["seats"] = X_test_cat["seats"].astype(str)

    cat_cols = [col for col in X_train_cat.columns if X_train_cat[col].dtype == "object"]

    combined = pd.concat([X_train_cat, X_test_cat], axis=0)

    encoded_combined = pd.get_dummies(combined[cat_cols], drop_first=True)

    encoded_X_train_cat = encoded_combined.iloc[:len(X_train_cat), :]
    encoded_X_test_cat = encoded_combined.iloc[len(X_train_cat):, :]

    final_X_train = pd.concat([X_train_cat[list(col for col in X_train_cat.columns if X_train_cat[col].dtype in ["int64", "float64"])], encoded_X_train_cat], axis=1)
    final_X_test = pd.concat([X_test_cat[list(col for col in X_test_cat.columns if X_test_cat[col].dtype in ["int64", "float64"])], encoded_X_test_cat], axis=1)

    return final_X_train, final_X_test

def process_name(value):
    data = value.split()
    return data[0], data[1]

def make_prediction(df):
    scaler = StandardScaler()

    df_tr_cat = df_train.copy()
    df_te_cat = df.copy()

    df_tr_cat[["brand", "model"]] = df_tr_cat.name.apply(lambda x: pd.Series(process_name(x)))
    df_te_cat[["brand", "model"]] = df_te_cat.name.apply(lambda x: pd.Series(process_name(x)))

    df_tr_cat = df_tr_cat.drop(columns=["name"])
    df_te_cat = df_te_cat.drop(columns=["name"])

    # final_X_train, final_X_test = proces_cat_features(df_tr_cat, df_te_cat)

    df_tr_cat["year"] = 2020 - df_tr_cat["year"]
    df_te_cat["year"] = 2020 - df_te_cat["year"]

    final_X_train, final_X_test = proces_cat_features(df_tr_cat, df_te_cat)

    num_cols = final_X_train.select_dtypes(include=["int64", "float64"]).columns
    cat_cols = final_X_train.select_dtypes(include=[bool]).columns

    poly = PolynomialFeatures(degree=2, include_bias=False)

    final_X_train_poly = pd.DataFrame(poly.fit_transform(final_X_train[num_cols]), columns=poly.get_feature_names_out(num_cols))
    final_X_test_poly = pd.DataFrame(poly.fit_transform(final_X_test[num_cols]), columns=poly.get_feature_names_out(num_cols))

    scaler.fit(final_X_train_poly)


    final_X_test_poly_transformed = pd.DataFrame(
        scaler.transform(final_X_test_poly),
        columns=final_X_test_poly.columns,
        index=final_X_test_poly.index
    )

    final_X_test = pd.concat([final_X_test[cat_cols], final_X_test_poly_transformed], axis=1)

    # Приводим признаки к единообразию
    missing = set(valid_cols) - set(final_X_test.columns)
    extra = set(final_X_test.columns) - set(valid_cols)

    # удаляем лишние
    final_X_test.drop(columns=list(extra), inplace=True)

    # добавляем недостающие
    for col in missing:
        final_X_test[col] = 0

    # порядок колонок
    final_X_test = final_X_test[valid_cols]

    y_pred = np.exp(model.predict(final_X_test))

    return (final_X_test, y_pred)

# Сразу загрузим модель и трейн датасет для визуализации EDA
with open("bestmodel.pkl", "rb") as f:
    model = pickle.load(f)

with open("columns.pkl", "rb") as f:
    valid_cols = pickle.load(f)

# Сразу загрузим модель и трейн датасет для визуализации EDA
with open("bestmodel.pkl", "rb") as f:
    model = pickle.load(f)

with open("cars_train.csv", "rb") as f:
    df_train = pd.read_csv("cars_train.csv")

df_train = data_preprocessing(df_train)

# Для примера выведем 3 графика
st.subheader("Фрагмент EDA по обучающим данным:")
plot_type = st.radio(
    "Тип графика:",
    ("Корреляционная матрица", "Jointplot по мощности и цене", "Boxplot цен по маркам")
)

if plot_type == "Корреляционная матрица":
    # Построение матрицы корреляций
    corr = df_train[[col for col in df_train.columns if df_train[col].dtype in ["int64", "float64"]]].corr()

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
    ax.set_title("Матрица корреляции Пирсона между числовыми признаками")

    st.pyplot(fig)

if plot_type == "Jointplot по мощности и цене":
    df_joint = df_train.copy()

    # Преобразуем столбцы
    df_joint["max_power"] = np.log(df_joint["max_power"])
    df_joint["selling_price"] = np.log(df_joint["selling_price"])

    # Построение Jointplot
    jp = sns.jointplot(
        data=df_joint, 
        x="max_power", 
        y="selling_price", 
        kind="kde",
        height=8
    )
    jp.set_axis_labels("Мощность", "Цена")
    jp.fig.suptitle("Jointplot по мощности и цене в log масштабе")

    # Отображение в Streamlit
    st.pyplot(jp.fig)

if plot_type == "Boxplot цен по маркам":

    df_brand = df_train.copy()
    df_brand["name"] = df_brand["name"].apply(lambda x: x.split()[0])

    # Берем топ-10 брендов
    top10 = df_brand["name"].value_counts().head(10).index
    df_top = df_brand[df_brand["name"].isin(top10)]

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_title("Распределение цен автомобилей по 10 популярным брендам")
    ax.set_xlabel("Бренд") 
    ax.set_ylabel("Цена")
    

    # Строим boxplot
    sns.boxplot(data=df_top, x="name", y="selling_price", ax=ax)

    st.pyplot(fig)

df = st.file_uploader("Загрузите csv файл", type=["csv"])


if df is not None:
    df = pd.read_csv(df)

    df = data_preprocessing(df)


    with st.expander("Показать датасет после предобработки"):
        st.dataframe(df)

    st.header("Параметры данных")
    col1, col2 = st.columns(2)

    with col1:
        st.metric("Количество столбцов", df.shape[1])
    
    with col2:
        st.metric("Количество строк", df.shape[0])

    final_X_test, y_pred = make_prediction(df)

    with st.expander("Показать Предсказания модели"):
        st.dataframe(y_pred)

    if "selling_price" in df.columns:
        y_test = df.selling_price

    r2 = r2_score(y_test, y_pred)
    Bus1 = business_metrics(y_test, y_pred)
    Bus2 = asymmetric_exp_error(y_test, y_pred)

    st.header("Метрики модели")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("R²:", f"{r2:.3f}")
    
    with col2:
        st.metric("Бизнес метрика 1:", f"{Bus1:.3f}")

    with col2:
        st.metric("Бизнес метрика 2:", f"{Bus2:.3f}")

    # Визуализация весов
    coef = model.coef_ 
    feature_names = final_X_test.columns 

    coef_df = pd.DataFrame({
        "Feature": feature_names,
        "Coefficient": coef
    }).sort_values(by="Coefficient", key=abs, ascending=False)

    st.subheader("Важность признаков (коэффициенты модели)")

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=coef_df.head(20), x="Coefficient", y="Feature", palette="viridis", ax=ax)
    ax.set_title("Топ 20 признаков по весам модели")
    st.pyplot(fig)

st.subheader("🔮 Предсказать стоимость автомобиля")

# Все исходные признаки, которые нужны модели
feature_names = df_train.columns

with st.form("manual_input_form"):
    st.write("Введите параметры автомобиля:")

    input_data = {}

    for col in feature_names:
        if df_train[col].dtype == "object":
            # Категориальные
            unique_vals = sorted(df_train[col].astype(str).unique().tolist())
            input_data[col] = st.selectbox(col, unique_vals)
        else:
            # Числовые
            default_val = float(df_train[col].median())
            input_data[col] = st.number_input(col, value=default_val)

    # 🔥 Обязательная кнопка отправки формы
    submitted = st.form_submit_button("Сделать предсказание", use_container_width=True)

if submitted:
    try:
        # Создаем датафрейм с одним объектом
        input_df = pd.DataFrame([input_data])

        # Предсказани
        y_pred = make_prediction(input_df)[1]

        st.success(f"**Предсказанная стоимость: {y_pred[0]:,.0f} $**")
    
    except Exception as e:
        st.error(f"Ошибка при предсказании: {e}")
