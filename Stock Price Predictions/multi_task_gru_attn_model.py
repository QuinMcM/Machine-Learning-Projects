import tensorflow as tf
import numpy as np
from tensorflow.keras import layers, Model
from tensorflow.keras.layers import Dense, GRU, Input, Dropout, MultiHeadAttention, Add, LayerNormalization
from tensorflow.keras.regularizers import l2
from tensorflow.keras.utils import register_keras_serializable
from tensorflow.keras.losses import Huber
from tensorflow.keras.callbacks import ReduceLROnPlateau
from sklearn.model_selection import train_test_split
import pickle
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# ============================================================================
# GPU CONFIGURATION
# ============================================================================

# Check GPU availability
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
print("GPU Details:", tf.config.list_physical_devices('GPU'))

# Enable memory growth to prevent TensorFlow from allocating all GPU memory at once
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"Memory growth enabled for {len(gpus)} GPU(s)")
    except RuntimeError as e:
        print(e)

# Set mixed precision for better GPU performance (optional but recommended)
# This can significantly speed up training on modern GPUs
tf.keras.mixed_precision.set_global_policy('mixed_float16')
print("Mixed precision enabled: mixed_float16")

# ============================================================================
# DATA LOADING
# ============================================================================

data = np.load('my_arrays.npz')

X_train = data['arr1']
y_train = data['arr2']

X_train = X_train[1:]
# y_train = y_train[1:]

# ============================================================================
# PART 1: ENHANCED TARGET PREPARATION
# ============================================================================

def prepare_multitask_targets_from_y(y_train_scaled, threshold=0.001):
    """
    Prepare multitask targets aligned with X_train using y_train (already aligned/scaled).

    Args:
        y_train_scaled (np.ndarray): The array of next-day prices for each input sequence.
        threshold (float): Minimum delta to label direction.

    Returns:
        dict: Aligned targets for multi-task output.
    """
    # Skip the first sample since we can't calculate a proper price change for it
    y_prices = y_train_scaled[1:].reshape(-1, 1)  # Current prices (excluding first)
    y_lagged = y_train_scaled[:-1].reshape(-1, 1)  # Previous prices (excluding last)

    # 1. Price target for custom loss
    price_combined = np.concatenate([y_prices, y_lagged], axis=1)

    # 2. Price change
    price_changes = y_prices - y_lagged

    # 3. Direction targets (Up=0, Down=1, Flat=2)
    direction_classes = np.where(
        price_changes > threshold, 0,
        np.where(price_changes < -threshold, 1, 2)
    ).astype(np.int32).flatten()

    direction_onehot = tf.keras.utils.to_categorical(direction_classes, num_classes=3)

    # 4. Magnitude targets
    magnitude = np.abs(price_changes)

    # ✅ Sanity print
    print("Shapes:")
    print(f"  y_prices: {y_prices.shape}")
    print(f"  price_combined: {price_combined.shape}")
    print(f"  direction_classes: {direction_classes.shape}")
    print(f"  direction_onehot: {direction_onehot.shape}")
    print(f"  magnitude: {magnitude.shape}")

    return {
        'price_combined': price_combined.astype(np.float32),  # Ensure float32 for GPU
        'direction_onehot': direction_onehot.astype(np.float32),
        'direction_classes': direction_classes,
        'magnitude': magnitude.astype(np.float32),
        'price_changes': price_changes.astype(np.float32)
    }


# ============================================================================
# PART 2: MULTI-OUTPUT MODEL ARCHITECTURE
# ============================================================================

def create_multitask_model(time_steps, num_features,
                          gru_units=128,
                          attention_heads=4,
                          attention_dim=64,
                          transformer_layers=4,
                          l2_reg=0.0):
    """
    Model with multiple output heads for multi-task learning
    """
    
    # Input layer
    ts_input = Input(shape=(time_steps, num_features), name="timeseries_input", dtype=tf.float32)
    
    # Shared backbone (your existing architecture)
    x = GRU(gru_units, return_sequences=True,
            kernel_regularizer=l2(l2_reg), dtype=tf.float32)(ts_input)
    x = Dropout(0.2)(x)
    
    # Transformer layers
    for i in range(transformer_layers):
        # Multi-Head Attention
        attn = MultiHeadAttention(num_heads=attention_heads, key_dim=attention_dim)(x, x)
        attn = Dropout(0.2)(attn)
        x = Add()([x, attn])
        x = LayerNormalization()(x)
        
        # Feedforward
        ff = Dense(x.shape[-1] * 4, activation='swish')(x)
        ff = Dropout(0.2)(ff)
        ff = Dense(x.shape[-1])(ff)
        x = Add()([x, ff])
        x = LayerNormalization()(x)
    
    # Shared features (last timestep)
    shared_features = x[:, -1, :]
    
    # ========================================================================
    # OUTPUT HEAD 1: PRICE PREDICTION (Your main task)
    # ========================================================================
    price_head = Dense(128, activation='swish', name="price_dense_1")(shared_features)
    price_head = Dropout(0.2)(price_head)
    price_head = Dense(32, activation='swish', name="price_dense_2")(price_head)
    # Cast to float32 for mixed precision
    price_output = Dense(1, name="price_output", dtype=tf.float32)(price_head)
    
    # ========================================================================
    # OUTPUT HEAD 2: DIRECTION CLASSIFICATION
    # ========================================================================
    direction_head = Dense(128, activation='swish', name="direction_dense_1")(shared_features)
    direction_head = Dropout(0.2)(direction_head)
    direction_head = Dense(32, activation='swish', name="direction_dense_2")(direction_head)
    direction_output = Dense(3, activation='softmax', name="direction_output", dtype=tf.float32)(direction_head)
    
    # ========================================================================
    # OUTPUT HEAD 3: MAGNITUDE PREDICTION
    # ========================================================================
    magnitude_head = Dense(128, activation='swish', name="magnitude_dense_1")(shared_features)
    magnitude_head = Dropout(0.2)(magnitude_head)
    magnitude_head = Dense(32, activation='swish', name="magnitude_dense_2")(magnitude_head)
    magnitude_output = Dense(1, activation='sigmoid', name="magnitude_output", dtype=tf.float32)(magnitude_head)
    
    # Create model with multiple outputs
    model = Model(
        inputs=ts_input, 
        outputs={
            'price_output': price_output,
            'direction_output': direction_output,
            'magnitude_output': magnitude_output
        }
    )
    
    return model

# ============================================================================
# PART 3: MULTI-TASK LOSS FUNCTION
# ============================================================================

@register_keras_serializable()
class EnhancedPriceLoss(tf.keras.losses.Loss):
    def __init__(self, beta=0.05, alpha=5.0, delta=0.05, name="enhanced_price_loss"):
        super().__init__(name=name)
        self.beta = beta
        self.alpha = alpha
        self.delta = delta
        self.huber_loss = tf.keras.losses.Huber(delta=delta)

    def call(self, y_true_combined, y_pred):
        # Ensure float32 for GPU computations
        y_true_combined = tf.cast(y_true_combined, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        
        y_true_price = y_true_combined[:, 0:1]
        y_lagged_price = y_true_combined[:, 1:2]
        
        price_huber = self.huber_loss(y_true_price, y_pred)
        
        true_delta = y_true_price - y_lagged_price
        pred_delta = y_pred - y_lagged_price
        lag_penalty = tf.reduce_mean(tf.square(tf.abs(true_delta) - tf.abs(pred_delta)))
        
        vec_pred = y_pred - y_lagged_price
        vec_true = y_true_price - y_lagged_price
        dot_product = vec_pred * vec_true
        norm_pred = tf.linalg.norm(vec_pred, axis=0) + 1e-6
        norm_true = tf.linalg.norm(vec_true, axis=0) + 1e-6
        directional_score = tf.reduce_mean(dot_product / (norm_pred * norm_true))
        directional_loss = 1 - directional_score
        
        return price_huber + self.beta * lag_penalty + self.alpha * directional_loss

    def get_config(self):
        return {
            "beta": self.beta,
            "alpha": self.alpha,
            "delta": self.delta,
            "name": self.name
        }

# ============================================================================
# PART 4: CUSTOM METRICS FOR MONITORING
# ============================================================================

class MultiTaskMetrics:
    """Container for all custom metrics"""
    
    @staticmethod
    def directional_accuracy_from_price():
        """Metric to track directional accuracy from price predictions"""
        def directional_accuracy(y_true_dict, y_pred_dict):
            price_combined = y_true_dict['price_output']
            price_pred = y_pred_dict['price_output']
            
            y_true_price = price_combined[:, 0:1]
            y_lagged_price = price_combined[:, 1:2]
            
            true_direction = tf.sign(y_true_price - y_lagged_price)
            pred_direction = tf.sign(price_pred - y_lagged_price)
            
            accuracy = tf.reduce_mean(tf.cast(tf.equal(true_direction, pred_direction), tf.float32))
            return accuracy
        
        return directional_accuracy
    
    @staticmethod
    def direction_classification_accuracy():
        """Standard accuracy for direction classification"""
        def direction_accuracy(y_true_dict, y_pred_dict):
            direction_true = y_true_dict['direction_output']
            direction_pred = y_pred_dict['direction_output']
            
            return tf.keras.metrics.categorical_accuracy(direction_true, direction_pred)
        
        return direction_accuracy

# ============================================================================
# PART 5: TRAINING SETUP EXAMPLE
# ============================================================================

def setup_multitask_training_alternative(X_train, prices, time_steps, num_features):
    """
    Alternative setup using per-output losses (cleaner approach)
    """
    
    # 1. Prepare targets
    aligned_prices = prices
    targets_dict = prepare_multitask_targets_from_y(aligned_prices, threshold=0.001)
    
    # 2. Create model
    model = create_multitask_model(time_steps, num_features)
    
    # 3. Setup optimizer
    initial_lr = 5e-5
        # lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        #     initial_learning_rate=initial_lr,
        #     decay_steps=5000,
        #     decay_rate=0.98
        # )
    reduce_lr = ReduceLROnPlateau(
        monitor='loss',       # metric to monitor (change to 'loss' if no validation set)
        factor=0.5,               # reduce LR by this factor (new_lr = old_lr * factor)
        patience=5,               # epochs to wait before reducing LR
        min_lr=1e-7,              # lower bound on LR
        verbose=1                 # print updates
    )
    optimizer = tf.keras.optimizers.Adam(learning_rate=initial_lr, clipnorm=1.0)
    
    # 4. Compile with separate losses per output
    model.compile(
        optimizer=optimizer,
        loss={
            'price_output': Huber(0.05),
            'direction_output': 'categorical_crossentropy',
            'magnitude_output': 'mse'
        },
        loss_weights={
            'price_output': 0.4,
            'direction_output': 3.0,
            'magnitude_output': 2.0
        },
        metrics={
            'price_output': ['mse', 'mae'],
            'direction_output': ['accuracy'],
            'magnitude_output': ['mse']
        },
        run_eagerly=False  # Critical for GPU performance
    )
    
    # 5. Prepare training data
    y_train_dict = {
        'price_output': targets_dict['price_combined'],
        'direction_output': targets_dict['direction_onehot'],
        'magnitude_output': targets_dict['magnitude']
    }
    
    print("Multi-task model setup complete (alternative approach)!")
    print(f"Model has {model.count_params()} parameters")
    
    return model, y_train_dict, targets_dict, reduce_lr

# ============================================================================
# MAIN EXECUTION
# ============================================================================

# Ensure input data is float32 for GPU
X_train = X_train.astype(np.float32)
y_train = y_train.astype(np.float32)

print(X_train.shape)
print(y_train.shape)

# Example usage:
model, y_train_dict, targets, reduce_lr = setup_multitask_training_alternative(X_train, y_train, 100, 46)

# Split X and y dict in a consistent way
X_train_split, X_val, _, _ = train_test_split(
    X_train, X_train, test_size=0.2, shuffle=False
)

y_train_split = {}
y_val_split = {}
for key, y in y_train_dict.items():
    y_train_split[key], y_val_split[key], _, _ = train_test_split(
        y, y, test_size=0.2, shuffle=False
    )

print(f"X_train shape: {X_train.shape}")

for name, arr in y_train_dict.items():
    if hasattr(arr, 'shape'):
        print(f"{name} shape: {arr.shape}")
    else:
        print(f"{name} is not a NumPy array or Tensor (type: {type(arr)})")

reduce_lr_callback = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=5,
    min_lr=1e-7,
    verbose=1
)

# Train model with explicit validation
print("\nStarting training on GPU...")
with tf.device('/GPU:0'):  # Explicitly place computation on GPU
    history = model.fit(
        X_train, 
        {
            'price_output': targets['price_combined'],
            'direction_output': targets['direction_onehot'],
            'magnitude_output': targets['magnitude'],
        },
        # validation_data=(X_val, y_val_split),
        batch_size=32,
        epochs=100,
        callbacks=[reduce_lr],
        verbose=2
    )

model.save("GRU Attention Model 4.keras")

with open('history.pkl', 'wb') as f:
    pickle.dump(history.history, f)

print("\nTraining complete!")