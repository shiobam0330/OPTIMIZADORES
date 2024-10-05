import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator

st.title("API Métodos de optimización")

if 'datos' not in st.session_state:
    st.session_state.datos = None 

if 'grafico1' not in st.session_state:
    st.session_state.grafico1 = None  

with st.container():
    col1, col2 = st.columns(2)

    with col1:
        st.header("Gráfico 3D de los datos")
        Inferior = st.number_input("Ingrese el límite inferior",  value=-6.5)
        Superior = st.number_input("Ingrese el límite superior",  min_value=(Inferior + 0.1), value=6.5)
        st.session_state.limite = (Inferior, Superior)

        if st.button("Graficar"):
            fig, ax = plt.subplots(subplot_kw={"projection": "3d"},figsize=(10,10))
            X = np.arange(Inferior, Superior, 0.25)
            Y = np.arange(Inferior, Superior, 0.25)
            X, Y = np.meshgrid(X, Y)
            R = np.sqrt(X**2 + Y**2)
            Z = -np.sin(R)
            
            surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                       linewidth=0, antialiased=True)
            ax.set_zlim(-1.01, 1.01)
            ax.zaxis.set_major_locator(LinearLocator(10))
            ax.zaxis.set_major_formatter('{x:.02f}')
            fig.colorbar(surf, shrink=0.5, aspect=5)
            st.session_state.grafico1 = fig  
        
        if st.session_state.grafico1 is not None:
            st.pyplot(st.session_state.grafico1)

    with col2:
        def evaluar_grad(x, y):
            R = np.sqrt(x**2 + y**2)
            grad_x = -np.cos(R) * (x / R)
            grad_y = -np.cos(R) * (y / R)
            return np.array([grad_x, grad_y])
        
        def gd(theta, epochs, eta):
            for i in range(epochs):
                x, y = theta
                gradient = evaluar_grad(x,y)
                theta -= eta * gradient
            return theta
        
        def sgd(theta, data_train, epochs, eta):
            for i in range(epochs):
                np.random.shuffle(data_train) 
                for example in data_train:
                    x, y = example
                    gradient = evaluar_grad(x, y)
                    theta = theta - eta * gradient 
            return theta
        
        def rmsprop(theta, data_train, epochs, eta, decay, epsilon):
            E_g2 = np.zeros_like(theta)
            for i in range(epochs):
                np.random.shuffle(data_train)
                for example in data_train:
                    x, y = example
                    gradient = evaluar_grad(x, y)
                    E_g2 = decay * E_g2 + (1 - decay) * gradient**2
                    theta -= eta / (np.sqrt(E_g2) + epsilon) * gradient
            return theta
        
        def adam(theta, data_train, epochs, alpha=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
            m = np.zeros_like(theta)
            v = np.zeros_like(theta)  
            t = 0  
            
            for epoch in range(epochs):
                np.random.shuffle(data_train) 
                for example in data_train:
                    x, y = example
                    t += 1 
                    gradient = evaluar_grad(x, y)
                    
                    m = beta1 * m + (1 - beta1) * gradient
                    v = beta2 * v + (1 - beta2) * (gradient**2)
                    
                    m_hat = m / (1 - beta1**t)
                    v_hat = v / (1 - beta2**t)
                    theta -= alpha * m_hat / (np.sqrt(v_hat) + epsilon)
            return theta

        st.header("Optimización")
        Inferior, Superior = st.session_state.limite
        Metodo = st.selectbox("Seleccione el optimizador a usar: ", ["","Gradient descent","Stochastic Gradient Descent","RMSPROP","Adam"])
        
        if Metodo == "Gradient descent":
            theta = st.number_input("Ingrese el theta inicial", value=2.0)
            tasa = st.number_input("Ingrese la tasa de aprendizaje", min_value=0.001, value=0.1)
            iter = st.number_input("Ingrese el número de iteraciones", min_value=1, value=1000)
            theta_1 = np.array([float(theta), float(theta)]) 

            if st.button("Calcular"):
                theta_final = gd(theta_1, iter, tasa)
                st.write(f"Punto estimado mínimo: {theta_final}") 

        elif Metodo == "Stochastic Gradient Descent":
            theta = st.number_input("Ingrese el theta inicial", value=2.0)
            tasa = st.number_input("Ingrese la tasa de aprendizaje", min_value=0.001, value=0.01)
            iter = st.number_input("Ingrese el número de iteraciones", min_value=1, value=100)
            n_puntos = st.number_input("Ingrese el número de puntos", min_value=1, value=100)
            theta_1 = np.array([float(theta), float(theta)]) 

            np.random.seed(1001300296)
            x_train = np.random.uniform(Inferior, Superior, n_puntos, )
            y_train = np.random.uniform(Inferior, Superior, n_puntos)
            data_train = list(zip(x_train, y_train))
            
            if st.button("Calcular"):
                 theta_final = sgd(theta_1, data_train, iter, tasa)
                 st.write(f"Punto estimado mínimo: {theta_final}") 

        elif Metodo == "RMSPROP":
            theta = st.number_input("Ingrese el theta inicial", value=2.0)
            tasa = st.number_input("Ingrese la tasa de aprendizaje", min_value=0.001, value=0.001)
            iter = st.number_input("Ingrese el número de iteraciones", min_value=1, value=100)
            decay = st.number_input("Ingrese el decay", min_value = 0.0, value =0.9)
            n_puntos = st.number_input("Ingrese el número de puntos", min_value=1, value=100)
            epsilon = st.number_input("Ingrese el epsilon", min_value=0.0, value=1e-8)
            theta_1 = np.array([float(theta), float(theta)]) 

            np.random.seed(1001300296)
            x_train = np.random.uniform(Inferior, Superior, n_puntos)
            y_train = np.random.uniform(Inferior, Superior, n_puntos)
            data_train = list(zip(x_train, y_train))
            
            if st.button("Calcular"):
                 theta_final = rmsprop(theta_1, data_train, iter, tasa, decay, epsilon)
                 st.write(f"Punto estimado mínimo: {theta_final}") 

        elif Metodo == "Adam":
            theta = st.number_input("Ingrese el theta inicial", value=2.0)
            iter = st.number_input("Ingrese el número de iteraciones", min_value=1, value=100)
            alpha = st.number_input("Ingrese el decay", min_value = 0.0, value =0.001)
            beta1 = st.number_input("Ingrese el epsilon", min_value=0.0, value=0.9)
            beta2 = st.number_input("Ingrese el epsilon", min_value=0.0, value=0.999)
            epsilon = st.number_input("Ingrese el epsilon", min_value=0.0, value=1e-8)
            n_puntos = st.number_input("Ingrese el número de puntos", min_value=1, value=100)
            theta_1 = np.array([float(theta), float(theta)]) 

            np.random.seed(1001300296)
            x_train = np.random.uniform(Inferior, Superior, n_puntos)
            y_train = np.random.uniform(Inferior, Superior, n_puntos)
            data_train = list(zip(x_train, y_train))
            
            if st.button("Calcular"):
                 theta_final = adam(theta_1, data_train, iter, alpha, beta1, beta2, epsilon)
                 st.write(f"Punto estimado mínimo: {theta_final}") 
