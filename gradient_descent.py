import numpy as np
import pandas as pd

def gradient_descent(x,y,lr=0.01,epochs=3000):
    m,b = 0.0, 0.0

    # Scale x and  y using min-max scaling
    x_scaled = (x - x.min()) / (x.max() - x.min())
    y_scaled = (y - y.min()) / (y.max() - y.min())

    for epoch in range(epochs):
        y_pred = m * x_scaled + b
        cost = np.mean((y_scaled - y_pred) ** 2)
        m_gradient = -2 * np.mean(x_scaled * (y_scaled - y_pred))
        b_gradient = -2 * np.mean(y_scaled - y_pred)
        m = m - lr * m_gradient
        b = b - lr * b_gradient

        if epoch % 100 == 0:
            print(f'm: {m}, b: {b}, cost: {cost} at epoch: {epoch}')

    # Scale back the coefficients
    m = m * (y.max() - y.min()) / (x.max() - x.min())
    b = b * (y.max() - y.min()) + y.min() - m * x.min()

    return m, b
    


if __name__ == '__main__':
    df = pd.read_csv('home_prices.csv')
    x = df['area_sqr_ft']
    y = df['price_lakhs']
    m, b = gradient_descent(x,y)
    print(f'Final result - m: {m}, b: {b}')