import numpy as np
import tensorflow as tf
from tqdm import trange

# Global Variables
MIN_CHANGE = 0.05                # Min change > 0.2 is not advised as the normalized values do not allow for predictions outside [0,1]
LEARNING_RATE = 1e-2
MAX_ITER = 300
BATCH_SIZE = 5                   # Batch sizes > 5 don't generate good results
INITIAL_CONST = 0.1
LARGEST_CONST = 100
SEQUENCE_LENGTH = 36
DECREASE_FACTOR = 0.9
VERBOSE = True

class CarliniLinf:
    """
    This is a modified version of the L_inf optimized attack of Carlini and Wagner (2016).
    It has been modified to fit time series regression problems.
    """
    def __init__(self, model, min_change = MIN_CHANGE, learning_rate = LEARNING_RATE,
                 max_iter = MAX_ITER, batch_size = BATCH_SIZE, initial_const = INITIAL_CONST,
                 largest_const = LARGEST_CONST, sequence_length = SEQUENCE_LENGTH,
                 decrease_factor = DECREASE_FACTOR, verbose = VERBOSE):
        """
        Create a Carlini&Wagner L_inf attack instance.
        :param model: A trained regressor model.
        :param min_change: The minimum change of the output that signals a successful attack.
        :param learning_rate: The initial learning rate for the attack algorithm. Smaller values produce better
                results but are slower to converge.
        :param max_iter: The maximum number of iterations.
        :param binary_search_steps: The number of times to adjust the constant with binary search.
        :param batch_size: Size of the batch on which adversarial samples are generated.
        :param initial_const: The initial value of the constant c.
        :param largest_const: The largest value for the constant c.
        :param sequence_length: The sequence length for the time series data.
        :param decrease_factor: The rate at which tau shrinks.
        :param verbose: Show progress bars.
        """

        self.model = model
        self.min_change = min_change
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.initial_const = initial_const
        self.largest_const = largest_const
        self.sequence_length = sequence_length
        self.decrease_factor = decrease_factor
        self.verbose = verbose

    def generate(self, x: np.ndarray, **kwargs) -> np.ndarray:
        """
        Perform the L_inf attack on the given time series data.
        :param x: An array with the original inputs to be attacked.
        :return: An array holding the adversarial examples.
        """
        num_rows, num_cols = x.shape
        
        # Reformat test set to include sequence length
        x_sequence = []
        for i in range(len(x)-self.sequence_length):
            s = []
            for j in range(0, self.sequence_length):
                s.append(x[[(i + j)], :])
            x_sequence.append(s)
        x_sequence = np.array(x_sequence)
        x_sequence = x_sequence.reshape(x_sequence.shape[0],self.sequence_length, num_cols)
        
        # Generate adversarial examples
        x_adv = np.zeros(x_sequence.shape)
        nb_batches = int(np.ceil(x_sequence.shape[0] / float(self.batch_size)))
        for i in trange(nb_batches, desc="C&W L_inf", disable = not self.verbose):
            index = i * self.batch_size
            x_adv[index:index+self.batch_size] = (self.generate_batch(x_sequence[index:index+self.batch_size]))
        print(x_adv.shape)
        return x_adv
    
    def generate_batch(self, x: np.ndarray, **kwargs) -> np.ndarray:
        """
        Generate batched adversarial samples and return them in an array.
        :param x: An array with the batched original inputs to be attacked.
        :return: An array holding the batched adversarial examples.
        """
        
        # Initialize placeholders for best l2 distance and attack found so far
        best_linf_dist = np.inf * np.ones(x.shape[0])
        best_x_adv = x.copy()
        
        pred = self.model.predict(x)
        
        # Initialize boolean to decide if advesarial examples should predict above or below original
        # Since the adv examples are normalized between [0,1], adv examples that predict values approaching 0 or 1 are difficult to generate, hence the bool
        mean = pred.mean()
        above = (mean <= .5)
        
        # Initialize variable to optimize
        w = tf.Variable(np.zeros(x.shape), trainable=True, dtype=tf.float32)
        best_w = w
        x_adv = (tf.tanh(w) + 1.0) / 2.0
        
        # Initialize variables to iterate over
        c_current = self.initial_const
        tau = np.max(np.abs(x_adv - x))

        while tau > 1./256:
            
            # Using "warm-start", meaning start gradient descent from the solution found on the previous iteration
            w = best_w

            found = False
            
            while c_current < self.largest_const and not found:
                
                for i_iter in range(self.max_iter):
    
                    # Calculate loss
                    with tf.GradientTape() as tape:
                        tape.watch(w)
                        
                        # Generate adversarial examples using w
                        x_adv = (tf.tanh(w) + 1.0) / 2.0
                        pred_adv = self.model(x_adv)
                        
                        # Calculate the first loss term using tau
                        tau_loss = tf.reduce_sum(tf.maximum(0.0, (tf.abs(x_adv - x) - tau)))
                        
                        # Loss depends if adv prediction is meant to be above or below the benign prediction
                        if above:
                            f_sum = tf.add(tf.add(pred, self.min_change), tf.negative(pred_adv))
                        else:
                            f_sum = tf.add(tf.add(tf.negative(pred), self.min_change), pred_adv)
                        c_loss = tf.multiply(c_current, tf.maximum(f_sum, tf.zeros(x_adv.shape[0])))
                        
                        # Add the two sums from the loss function
                        loss = tf.add(tau_loss, c_loss)
                    
                    # Calculate loss gradient w.r.t our optimization variable w 
                    gradients = tape.gradient(loss, w)
                    
                    # Update w
                    w = tf.subtract(w, tf.multiply(self.learning_rate, gradients))
                    
                    # Calculate linf_dist and generate new adversarial predictions
                    x_adv = x_adv.numpy()
                    pred_adv = self.model.predict(x_adv)
                    linf_dist = np.max(np.abs(x_adv - x).reshape(x.shape[0], -1), axis=1)
                    
                    # Update adversarial examples if new best is found
                    for e in range(x.shape[0]): 
                        if above:
                            if pred_adv[e] >= pred[e] + self.min_change and linf_dist[e] < best_linf_dist[e]:
                                best_x_adv[e] = x_adv[e]
                                best_linf_dist[e] = linf_dist[e]
                                best_w = w
                                found = True
                        else:
                            if pred_adv[e] <= pred[e] - self.min_change and linf_dist[e] < best_linf_dist[e]:
                                best_x_adv[e] = x_adv[e]
                                best_linf_dist[e] = linf_dist[e]
                                best_w = w
                                found = True
                
                    # Early abort if there are no gradient updates
                    if tf.reduce_sum(gradients) == 0:
                        break
                    
                # Update constant c if no example has been found yet
                if not found:
                    c_current *= 2
            
            tau *= self.decrease_factor
            
        return best_x_adv