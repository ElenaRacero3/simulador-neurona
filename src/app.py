
from neurona import Neuron
import streamlit as st


# Streamlit app
          
st.set_page_config(layout="wide")
st.header("Simulador de neurona")
st.image("img/neurona.png", width = 400)
st.write("Elige el número de entradas/pesos que tendrá la neurona")
w = st.slider('Peso', 1, 10, key = 1)

st.subheader("Pesos")

columna1  = st.columns(w)
weights = []
for i in range(w):
   with columna1[i]:
      weights.append(st.number_input(f'w{i}'))

st.text(f"w = {weights}")

st.subheader("Entradas")

columna2  = st.columns(w)
inputs = []
for i in range(w):
   with columna2[i]:
      inputs.append(st.number_input(f'x{i}'))

st.text(f"x = {inputs}")

col1, col2 = st.columns(2)
with col1:
   st.subheader("Sesgo")
   bias = st.number_input('Introduce el valor del sesgo', key = 3)

with col2:
   st.subheader("Función de activación")
   func = st.selectbox(
    'Elige la funación de activación',
    ('Tangente hiperbólica', 'Sigmoide', 'ReLU'), key = 4)

if st.button('Calcular salida', key = 7):
    n = Neuron(weights=weights, bias=bias, func=func.lower())
    output = n.run(input_data=inputs)
    st.write(f"La salida de la neurona es {output}")

