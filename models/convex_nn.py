import tensorflow as tf 
import pickle
import os 
import numpy as np 


class AbsInitializer(tf.keras.initializers.Initializer):

  def __init__(self, original_initializer):
    self.origin_init = original_initializer

  def __call__(self, shape, dtype=None, **kwargs):
    return self.origin_init(
        shape,dtype=dtype, **kwargs)

  def get_config(self):  
    return self.origin_init.get_config()


class ResidualPartialConvexLayer(tf.keras.layers.Layer):
    def __init__(self, units, activation=None, recursive_convexity=False,  kernel_initializer='glorot_uniform'):
        super(ResidualPartialConvexLayer, self).__init__()
        self.units = units
        self.activation = activation
        self.recursive_convexity = recursive_convexity
        self.kernel_initializer = tf.keras.initializers.get(kernel_initializer)
        self.positive_kernel_initializer = AbsInitializer(self.kernel_initializer)


    def build(self, input_shapes):
        input_dim_z, input_dim_x, input_dim_u = input_shapes[0][-1], input_shapes[1][-1], input_shapes[2][-1]

        self.kernel_z = self.add_weight("kernel_z", shape=[input_dim_z, self.units],  initializer=self.positive_kernel_initializer)
        self.kernel_x = self.add_weight("kernel_x", shape=[input_dim_x, self.units],  initializer=self.kernel_initializer if not self.recursive_convexity else self.positive_kernel_initializer)
        self.kernel_u = self.add_weight("kernel_u", shape=[input_dim_u, self.units],  initializer=self.kernel_initializer)

        self.kernel_zu = self.add_weight("kernel_zu", shape=[input_dim_u, input_dim_z],  initializer=self.kernel_initializer)
        self.kernel_xu = self.add_weight("kernel_xu", shape=[input_dim_u, input_dim_x],  initializer=self.kernel_initializer)

        self.bias = self.add_weight("bias", shape=[self.units])
        self.bias_z = self.add_weight("bias_z", shape=[input_dim_z], initializer="ones")
        self.bias_x = self.add_weight("bias_x", shape=[input_dim_x], initializer="ones")

    def get_config(self):
        config = super().get_config()
        config.update({'units':self.units, 'activation':self.activation, 'recursive_convexity':self.recursive_convexity,
                       'kernel_initializer': self.kernel_initializer})
        return config

    @classmethod
    def from_config(cls, **config):
        return cls(**config)


    def get_positive_weight(self):
        return [self.kernel_z, ] if not self.recursive_convexity else [self.kernel_z, self.kernel_x]

    def call(self, inputs):
        z_input, x_input, u_input = inputs

        gamma_z = tf.nn.relu(tf.matmul( u_input, self.kernel_zu) + self.bias_z)
        gamma_x = tf.matmul(u_input, self.kernel_xu) + self.bias_x

        output = tf.matmul(z_input*gamma_z, self.kernel_z)
        output= tf.add(output, tf.matmul(x_input*gamma_x, self.kernel_x))
        output = tf.add(output, tf.matmul(u_input, self.kernel_u))
        output = tf.add(output, self.bias)
        
        if self.activation is not None:
            output = self.activation(output)
        return output


class ResidualConvexLayer(tf.keras.layers.Layer):
    def __init__(self, units, activation=None, recursive_convexity=False, kernel_initializer='glorot_uniform', use_bias=True):
        super(ResidualConvexLayer, self).__init__()
        self.units = units
        self.activation = activation
        self.use_bias = use_bias
        self.recursive_convexity = recursive_convexity
        self.kernel_initializer = tf.keras.initializers.get(kernel_initializer)
        self.positive_kernel_initializer = AbsInitializer(self.kernel_initializer)


    def build(self, input_shapes):
        input_dim_z, input_dim_x = input_shapes[0][-1], input_shapes[1][-1]
        self.kernel_z = self.add_weight("kernel_z", shape=[input_dim_z, self.units],  initializer=self.positive_kernel_initializer)
        self.kernel_x = self.add_weight("kernel_x", shape=[input_dim_x, self.units],  initializer=self.kernel_initializer if not self.recursive_convexity else self.positive_kernel_initializer)

        if self.use_bias:
            self.bias = self.add_weight("bias", shape=[self.units])
        else:
            self.bias = None

    def get_positive_weight(self):
        return [self.kernel_z, ] if not self.recursive_convexity else [self.kernel_z, self.kernel_x]

    def call(self, inputs):
        z_input, x_input = inputs
        output = tf.matmul(z_input, self.kernel_z)
        output = tf.add(output, tf.matmul(x_input, self.kernel_x))

        if self.use_bias:
            output = tf.add(output, self.bias)
        
        if self.activation is not None:
            output = self.activation(output)
        print(output)
        return output


class ICNN():
    def __init__(self, nb_input=2, nb_output=1, act_func=tf.nn.relu, nb_layer=2, units=24, recursive_convexity=False,   kernel_initializer=tf.keras.initializers.GlorotUniform()):
        self.output_func = tf.identity
        self.act_func = act_func
        self.nb_layer = nb_layer
        self.kernel_initializer = kernel_initializer
        self.units = units
        self.recursive_convexity = recursive_convexity

        self.input_layer = tf.keras.layers.Input(shape=(nb_input,))

        self.z_layers = list()
        curr_z = tf.zeros_like(self.input_layer)
        self.strictly_positive_weight  = list()


        for i in range(self.nb_layer):
            layer = ResidualConvexLayer(self.units, activation=self.act_func, recursive_convexity=self.recursive_convexity, kernel_initializer=self.kernel_initializer)
            self.z_layers.append(layer)
            curr_z = layer([curr_z, self.input_layer])
            self.strictly_positive_weight.extend(layer.get_positive_weight())
        
        self.output_layer = ResidualConvexLayer(nb_output, activation=self.output_func, recursive_convexity=self.recursive_convexity, kernel_initializer=self.kernel_initializer)
        output = self.output_layer([curr_z, self.input_layer])
        self.strictly_positive_weight.extend(self.output_layer.get_positive_weight())

        self.core = tf.keras.Model(inputs=self.input_layer, outputs=output)
        self.core.build(input_shape=(nb_input,))

    @tf.function
    def compute_loss(self, y, x):
        y_hat = self.core(x)
        return tf.reduce_mean(tf.square(y_hat - y)) 

    def train(self, y, x, nb_epoch= 20,optimizer = tf.keras.optimizers.Adam(learning_rate=0.1)):

        x_tf = tf.constant(x, dtype=tf.float32)
        y_tf = tf.constant(y, dtype=tf.float32)

        trainable_variables = self.core.trainable_variables

        for epoch in range(nb_epoch):      

            with tf.GradientTape() as tape:  
                loss_val = self.compute_loss(y_tf, x_tf)

            # Apply gradient optimization
            gradients = tape.gradient(loss_val, trainable_variables)
            optimizer.apply_gradients(zip(gradients, trainable_variables))

            # Clip weight
            for weight in self.strictly_positive_weight:
                weight.assign(tf.math.maximum(weight, tf.zeros_like(weight)))
            

class PICNN():

    @classmethod
    def load(cls, path):
        conf = pickle.load(open(os.path.join(path, "obj.pickle"), "rb"))
        result = cls(nb_input=conf.nb_input, nb_input_conv=conf.nb_input_conv, nb_output=conf.nb_output, act_func=conf.act_func, nb_layer=conf.nb_layer, units=conf.units, recursive_convexity=conf.recursive_convexity, kernel_initializer=conf.kernel_initializer)
        result.core.load_weights(os.path.join(path, "model.h5"))
        return result

    # The fonction is convex for the first nb_input_conv layer
    def __init__(self, nb_input=2, nb_input_conv=2, nb_output=1, act_func=tf.nn.relu, nb_layer=2, units=24, recursive_convexity=False, kernel_initializer=tf.keras.initializers.GlorotUniform()):
        assert nb_input>=nb_input_conv, "Please provide a nb conv input equals or less than the total input number"
        
        self.output_func = tf.identity
        self.act_func = act_func
        self.nb_layer = nb_layer
        self.kernel_initializer = kernel_initializer
        self.units = units
        self.nb_output = nb_output
        self.nb_input = nb_input
        self.nb_input_conv = nb_input_conv
        self.recursive_convexity = recursive_convexity

        self.input_layer = tf.keras.layers.Input(shape=(nb_input,))

        self.x_input = tf.keras.layers.Lambda(lambda x: x[:,0:self.nb_input_conv])(self.input_layer)
        self.u_input = tf.keras.layers.Lambda(lambda x: x[:,self.nb_input_conv:])(self.input_layer)
        
        self.z_layers = list()
        self.u_layers = list()

        curr_z = tf.zeros_like(self.input_layer)
        curr_u = self.u_input

        self.strictly_positive_weight  = list()


        for i in range(self.nb_layer):
            layer = ResidualPartialConvexLayer(self.units, activation=self.act_func, recursive_convexity=self.recursive_convexity, kernel_initializer=self.kernel_initializer)
            self.z_layers.append(layer)
            curr_z = layer([curr_z, self.x_input, curr_u])
            layer_u = tf.keras.layers.Dense(self.units, activation=self.act_func, kernel_initializer=self.kernel_initializer)
            curr_u = layer_u(curr_u)
            self.u_layers.append(layer_u)
            self.strictly_positive_weight.extend(layer.get_positive_weight())
        
        self.output_layer = ResidualPartialConvexLayer(nb_output, activation=self.output_func, recursive_convexity=self.recursive_convexity, kernel_initializer=self.kernel_initializer)

        output = self.output_layer([curr_z, self.x_input, curr_u])

        self.strictly_positive_weight.extend(self.output_layer.get_positive_weight())

        self.core = tf.keras.Model(inputs=self.input_layer, outputs=output)

        self.core.build(input_shape=(nb_input,))

    def save(self, path):
        
        if not os.path.isdir(path):
            if os.path.isfile(path):
                raise RuntimeError("Bad path")

            os.mkdir(path)
        
        pickle.dump(self, open(os.path.join(path, "obj.pickle"), "wb"))
        self.core.save_weights(os.path.join(path, "model.h5"))

    def __getstate__(self):
        state = self.__dict__.copy()
        if 'core' in state:
            del state['core']
        if 'x_input' in state:
            del state['x_input']
        if 'u_input' in state:
            del state['u_input']
        if 'z_layers' in state:
            del state['z_layers']
        if 'u_layers' in state:
            del state['u_layers']        
        if 'strictly_positive_weight' in state:
            del state['strictly_positive_weight']        
        if 'input_layer' in state:
            del state['input_layer']     


        if 'output_layer' in state:
            del state['output_layer']      

        return state


    @tf.function
    def compute_loss(self, y, x):
        y_hat = self.core(x)
        return tf.reduce_mean(tf.square(y_hat - y)) 

    def train(self, y, x, nb_epoch=20, optimizer=tf.keras.optimizers.Adam(learning_rate=0.1)):

        x_tf = tf.constant(x, dtype=tf.float32)
        y_tf = tf.constant(y, dtype=tf.float32)

        trainable_variables = self.core.trainable_variables

        for epoch in range(nb_epoch):      

            with tf.GradientTape() as tape:  
                loss_val = self.compute_loss(y_tf, x_tf)

            # Apply gradient optimization
            gradients = tape.gradient(loss_val, trainable_variables)
            optimizer.apply_gradients(zip(gradients, trainable_variables))

            # Clip weight
            for weight in self.strictly_positive_weight:
                weight.assign(tf.math.maximum(weight, tf.zeros_like(weight)))
            
            print(loss_val)


