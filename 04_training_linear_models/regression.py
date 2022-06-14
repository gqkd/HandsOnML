#%%
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mlt
from sklearn.linear_model import LinearRegression
#regression y_cap = theta0 + theta1*x1 + ... + thetaN*xn
# also y_cap = theta * x , where theta and x are vectors and * is the dot product
#we can train using RMSE or MSE, MSE minimization is easier

#we want to find theta that minimize the cost function, there is a closed form solution
# theta_cap = (X^T*X)^-1 * X^T * y
# this is the normal equation
#%%
#lets create a distribution

# random distribution
X = 2 * np.random.rand(100, 1)
# normal distribution
y = 4 + 3 * X + np.random.randn(100, 1)

plt.plot(X,y,"b.")
plt.ylabel("$y$", rotation=0, fontsize=18)
plt.xlabel("$x_1$", rotation=0, fontsize=18)
plt.axis([0,2,0,15])

#lets compute theta_cap with normal equation
X_b = np.c_[np.ones((100, 1)), X] #add x0 = 1 on the first column
theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)

print(theta_best)

#lets create the new point in the regression
X_new = np.array([[0],[2]])
X_new_b = np.c_[np.ones((2,1)), X_new] 

y_predict = X_new_b.dot(theta_best)
print(y_predict)

plt.plot(X_new, y_predict, "r-")
plt.plot(X, y, "b.")
plt.axis([0,2,0,15])
plt.show()

# %% linear regressione with scikit-learn

#scikit compute the pseudo inverse (moore-penrose)
# theta_cap = X(pseudoinv) * y
lin_reg = LinearRegression()
lin_reg.fit(X, y)
print(lin_reg.intercept_, lin_reg.coef_)
print(lin_reg.predict(X_new))

theta_best_svd, residuals, rank, s = np.linalg.lstsq(X_b, y, rcond=1e-6)
print(theta_best_svd)

print(np.linalg.pinv(X_b).dot(y))


# %% gradient descendent

#gradient descendent is an optimization algorithm, it measure the local gradient of the error function
# with regard to the parameter theta and it goes in descent direction
# the size of the step are fundamental hyperparameters, also called learning rate
# too small, it takes long time, too high, can increase theta
# we want to find a global minimum not a local one, the function can be not so regular
# MSE is a convex function, it has only a global minimum

#batch gradiente descendent
#the gradient of the cost function must be computed regard to all parameters theta_j
# to find how cost function change regard to changing of one parameters means compute the partial derivative
#instead of computing all the partial derivatives, you can compute the gradient vector
# gradient_vector = 2/m * X^T * (X*theta - y)
# you have to consider all X everytime, from that the name batch GD
# at this point you have the gradient vector, you must decide where is uphill and go downhill
# you have to subtract gradient_vector from actualt theta
# theta_next = theta - learning_rate * gradient_vector

#quick implementation of this algo
eta = 0.1
n_iterations = 1000
m = 100
#random initialization
theta = np.random.randn(2,1)

for iteration in range(n_iterations):
    gradients = 2/m * X_b.T.dot(X_b.dot(theta) - y)
    theta = theta - eta * gradients

print(theta)
#it works fine, lets plot something

theta_path_bgd = []

def plot_gradient_descent(theta, eta, theta_path=None):
    m = len(X_b)
    plt.plot(X, y, "g.")
    n_iterations = 1000
    for iteration in range(n_iterations):
        if iteration < 10:
            y_predict = X_new_b.dot(theta)
            style = "b-" if iteration > 0 else "r--"
            plt.plot(X_new, y_predict, style)
        gradients = 2/m * X_b.T.dot(X_b.dot(theta) - y)
        theta = theta - eta * gradients
        if theta_path is not None:
            theta_path.append(theta)
    plt.xlabel("$x_1$", fontsize=18)
    plt.axis([0, 2, 0, 15])
    plt.title(r"$\eta = {}$".format(eta), fontsize=16)

np.random.seed(42)
theta = np.random.randn(2,1)  # random initialization

plt.figure(figsize=(10,4))
plt.subplot(131); plot_gradient_descent(theta, eta=0.02)
plt.ylabel("$y$", rotation=0, fontsize=18)
plt.subplot(132); plot_gradient_descent(theta, eta=0.1, theta_path=theta_path_bgd)
plt.subplot(133); plot_gradient_descent(theta, eta=0.5)

plt.show()
# %% stochastic GD

