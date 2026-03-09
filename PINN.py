import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

class PINNmodel(tf.keras.Model):
    def __init__(self, layers=None, wtrain=1, activation='swish', **kwargs):
        super().__init__(**kwargs)
        self.layers_config = layers
        self.hidden = [tf.keras.layers.Dense(w, activation=activation) for w in layers[:-1]]
        self.out = tf.keras.layers.Dense(layers[-1])
        self.wtrain=wtrain
        self.s_phys = tf.Variable(0.0, trainable=1, dtype=tf.float32, name='s_phys')
        self.s_ic = tf.Variable(0.0, trainable=1, dtype=tf.float32, name='s_ic')
        self.s_l = tf.Variable(6.0, trainable=1, dtype=tf.float32, name='s_l')
    def call(self, t):
        x = t
        for layer in self.hidden:
            x = layer(x)
        return self.out(x)
    
    def get_config(self):
        config = super().get_config()
        config.update({"layers": self.layers_config})
        return config

    @classmethod
    def from_config(cls, config):
        layers = config.pop("layers", None)
        return cls(layers=layers, **config)
    
    def pinn_loss(self, t_phys, t0, u0, du0, m, c, k):
        with tf.GradientTape(persistent=True) as t2:
            t2.watch(t_phys)
            with tf.GradientTape() as t1:
                t1.watch(t_phys)
                u = self(t_phys)
            du_dt = t1.gradient(u, t_phys)
        d2u = t2.gradient(du_dt, t_phys)
        del t2
        res = m * d2u + c * du_dt + k * u
        loss_phys = tf.reduce_mean(tf.square(res))
        with tf.GradientTape() as t_ic:
            t_ic.watch(t0)
            u_pred0 = self(t0)
        du_pred0 = t_ic.gradient(u_pred0, t0)
        loss_ic =  tf.reduce_mean(tf.square(u_pred0 - u0)) + tf.reduce_mean(tf.square(du_pred0 - du0))
        if self.wtrain==1:
            s_phys = self.s_phys
            s_ic   = self.s_ic
            s_l = self.s_l
            # w_phys = tf.exp(s_phys)
            w_phys = s_l**s_phys
            # w_ic   = tf.exp(s_ic)
            w_ic = s_l**s_ic
            # w_l = tf.exp(s_l)
            total_w = w_phys + w_ic
            w_phys = w_phys / total_w * s_l
            w_ic = w_ic / total_w * s_l
            return w_phys*loss_phys + w_ic*loss_ic ,w_phys ,w_ic ,s_l
        else:
            w_phys=1.5
            w_ic=2.5
            return w_phys*loss_phys + w_ic*loss_ic ,tf.convert_to_tensor(w_phys) ,tf.convert_to_tensor(w_ic)

    def train_and_log(self, t_phys, t0, u0, du0, m, c, k, t_test, epochs=200, lr=1e-3):
        history = []
        optimizer = tf.keras.optimizers.Adam(lr)
        for epoch in range(epochs+1):
            # optimizer = tf.keras.optimizers.Adam(lr)
            with tf.GradientTape() as tape:
                loss ,w_phys ,w_ic ,s_l = self.pinn_loss(t_phys, t0, u0, du0, m, c, k)
            if self.wtrain==1:
                grads = tape.gradient(loss,self.trainable_variables+ [self.s_phys, self.s_ic ,self.s_l])
                optimizer.apply_gradients(zip(grads, self.trainable_variables+ [self.s_phys, self.s_ic ,self.s_l]))
            else:
                grads = tape.gradient(loss,self.trainable_variables)
                optimizer.apply_gradients(zip(grads, self.trainable_variables))
            # if epoch==3000:
            #     self.wtrain=1
            #     optimizer = tf.keras.optimizers.Adam(lr)
            if epoch%10 == 0:
                u_pred = self(t_test).numpy().flatten()
                history.append(u_pred)
            if epoch % 500 == 0:
                print(
                f"Epoch {epoch}: Loss={loss.numpy():.3e}, "
                f"w_phys={w_phys.numpy():.3f}, w_ic={w_ic.numpy():.3f}, "
                f"s_l={s_l.numpy():.3f}"
                )
        self.save('PINNmodel.keras')
        return np.array(history)

if __name__=="__main__":
    m, c, k = 1.0, 0.5, 2.0
    t_phys = tf.random.uniform((400,1), 0, 20, dtype=tf.float32)
    t0 = tf.constant([[0.0]], dtype=tf.float32)
    u0, du0 = tf.constant([[2.0]]), tf.constant([[2.0]])
    t_test = tf.linspace(0,20,400).numpy().reshape(400,1)

    model = PINNmodel([30,15,1],wtrain=1,activation='swish')
    history = model.train_and_log(t_phys, t0, u0, du0, m, c, k, t_test,epochs=12000, lr=1e-3)
    model.summary()

    u0=u0.numpy()
    du0=du0.numpy()
    omega0 = np.sqrt(k/m)
    zeta = c/(2*np.sqrt(m*k))
    omega_d = omega0 * np.sqrt(max(0,1 - zeta**2))
    # u_ex = u0 * np.exp(-zeta*omega0*t_test.flatten()) * np.cos(omega_d * t_test.flatten())
    u_ex = (u0 * np.cos(omega_d * t_test.flatten()) + (du0 + zeta * omega0 * u0) / omega_d * np.sin(omega_d * t_test.flatten())) * np.exp(-zeta * omega0 * t_test.flatten())


    fig, ax = plt.subplots()
    line_pred, = ax.plot([], [], 'b-', label='PINN')
    line_ex, = ax.plot(t_test, u_ex.reshape(400,1), '--', label='Analytical')
    ax.set_xlim(0, 20)
    ax.set_ylim(np.min(u_ex)-0.2, np.max(u_ex)+0.2)
    ax.legend()

    def animate(i):
        line_pred.set_data(t_test, history[i])
        ax.set_title(f"Epoch {i*10}")
        return line_pred,ax
    ani = FuncAnimation(fig, animate, frames=len(history), interval=10, blit=False)
    plt.show()
    plt.close()

# from PINN import PINNmodel
# model = tf.keras.models.load_model(
#     'PINNmodel.keras',
#     custom_objects={'PINNmodel': PINNmodel},
#     compile=False
# )
# t_test = tf.linspace(0,20,400).numpy().reshape(400,1)
# predictions = model.predict(t_test).flatten()
# plt.plot(t_test.flatten(), predictions, '-', label='PINN')
# plt.legend()
# plt.show()

