import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from PINN import PINNmodel

m, c, k = 1.0, 0.5, 2.0
t_phys = tf.random.uniform((400,1), 0, 20, dtype=tf.float32)
t0 = tf.constant([[0.0]], dtype=tf.float32)
u0, du0 = tf.constant([[2.0]]), tf.constant([[2.0]])
t_test = tf.linspace(0,20,400).numpy().reshape(400,1)

u0=u0.numpy()
du0=du0.numpy()
omega0 = np.sqrt(k/m)
zeta = c/(2*np.sqrt(m*k))
omega_d = omega0 * np.sqrt(max(0,1 - zeta**2))
u_ex = (u0 * np.cos(omega_d * t_test.flatten()) + (du0 + zeta * omega0 * u0) / omega_d * np.sin(omega_d * t_test.flatten())) * np.exp(-zeta * omega0 * t_test.flatten())

model1 = tf.keras.models.load_model('PINNmodeltanhw.keras', custom_objects={'PINNmodel': PINNmodel}, compile=False)
# model.summary()
plt.plot(t_test.flatten(), u_ex.reshape(400,1), '--', label='Analytical')
pred1 = model1.predict(t_test).flatten()
plt.plot(t_test.flatten(), pred1, '-', label='PINNw')
 
model2 = tf.keras.models.load_model('PINNmodelswish.keras', custom_objects={'PINNmodel': PINNmodel}, compile=False)
pred2 = model2.predict(t_test).flatten()
plt.plot(t_test.flatten(), pred2, '-', label='PINNn')
plt.legend()
plt.show()