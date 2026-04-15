import pickle
import sys
import time
import h5py
import numpy as np
import tensorflow as tf
import os
import glob

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"✅ GPU memory growth enabled for {len(gpus)} GPU(s)")
    except RuntimeError as e:
        print(f"❌ GPU memory growth setup failed: {e}")



def check_file_exists(file_path):

    if os.path.exists(file_path) == False:
        print("Error: provided file path '%s' does not exist!" % file_path)
        sys.exit(-1)
    return



def load_ascad(ascad_database_file, load_metadata=False, validation_split=5000, load_attack=True):

    check_file_exists(ascad_database_file)

    try:
        in_file = h5py.File(ascad_database_file, "r")
    except:
        print("Error: can't open HDF5 file '%s' for reading" % ascad_database_file)
        sys.exit(-1)

    # 加载训练集
    X_profiling = np.array(in_file['Profiling_traces/traces'], dtype=np.float64)
    Y_profiling = np.array(in_file['Profiling_traces/labels'])


    if not load_attack:
        if validation_split > 0:


            X_validation = np.array(in_file['Attack_traces/traces'][:validation_split], dtype=np.float64)
            Y_validation = np.array(in_file['Attack_traces/labels'][:validation_split])

            print(f"  Validation set: {X_validation.shape[0]} 条")


            in_file.close()

            if load_metadata:
                return (X_profiling, Y_profiling), (X_validation, Y_validation), None, None
            else:
                return (X_profiling, Y_profiling), (X_validation, Y_validation), (None, None)
        else:

            print(f"  Training set: {X_profiling.shape[0]} 条")


            in_file.close()

            if load_metadata:
                return (X_profiling, Y_profiling), None, None
            else:
                return (X_profiling, Y_profiling), (None, None)


    print(f"\n Dataset split (evaluation mode - load complete data):")
    X_attack_full = np.array(in_file['Attack_traces/traces'], dtype=np.float64)
    Y_attack_full = np.array(in_file['Attack_traces/labels'])


    if validation_split > 0:

        X_validation = X_attack_full[:validation_split]
        Y_validation = Y_attack_full[:validation_split]


        X_attack = X_attack_full[validation_split:]
        Y_attack = Y_attack_full[validation_split:]

        print(f"  Training set: {X_profiling.shape[0]} samples")
        print(f"  Validation set: {X_validation.shape[0]} samples (first {validation_split} samples from Attack)")
        print(
            f"  Test set: {X_attack.shape[0]} samples (last {len(Y_attack_full) - validation_split} samples from Attack)")
        if load_metadata:
            metadata_profiling = in_file['Profiling_traces/metadata']
            metadata_attack_full = np.array(in_file['Attack_traces/metadata'])
            metadata_validation = metadata_attack_full[:validation_split]
            metadata_attack = metadata_attack_full[validation_split:]
            return (X_profiling, Y_profiling), (X_validation, Y_validation), (X_attack, Y_attack), \
                   (metadata_profiling, metadata_validation, metadata_attack)
        else:
            return (X_profiling, Y_profiling), (X_validation, Y_validation), (X_attack, Y_attack)
    else:

        if load_metadata:
            return (X_profiling, Y_profiling), (X_attack_full, Y_attack_full), \
                   (in_file['Profiling_traces/metadata'], in_file['Attack_traces/metadata'])
        else:
            return (X_profiling, Y_profiling), (X_attack_full, Y_attack_full)


# ==================== APS downsampling function ====================
def APS_downsample_1D(x, stride=2, p_norm=2):
    """Adaptive Polyphase Sampling for 1D signals"""
    input_length = tf.shape(x)[1]
    output_length = tf.cast(tf.math.floor(tf.cast(input_length, tf.float32) / stride), tf.int32)

    polyphase_components = []
    norms = []

    for i in range(stride):
        component = x[:, i::stride, :]
        component = component[:, :output_length, :]
        polyphase_components.append(component)

        if p_norm == 2:
            norm = tf.reduce_sum(tf.square(component), axis=[1, 2])
        else:
            norm = tf.reduce_sum(tf.pow(tf.abs(component), p_norm), axis=[1, 2])
        norms.append(norm)

    norms_stack = tf.stack(norms, axis=0)
    max_indices = tf.argmax(norms_stack, axis=0)
    batch_size = tf.shape(x)[0]
    batch_indices = tf.range(batch_size, dtype=tf.int64)
    components_tensor = tf.stack(polyphase_components, axis=0)
    components_tensor = tf.transpose(components_tensor, [1, 0, 2, 3])
    indices = tf.stack([batch_indices, max_indices], axis=1)
    selected = tf.gather_nd(components_tensor, indices)

    return selected


# ==================== Basic convolution block ====================
def Conv_1D_Block(x, model_width, kernel, strides=1, padding="same"):

    x = tf.keras.layers.Conv1D(
        model_width, kernel, strides=strides,
        padding=padding, kernel_initializer="he_normal"
    )(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('selu')(x)
    return x


def Conv_1D_Block_Dilated(x, model_width, kernel, dilation_rate=1, padding="same"):

    x = tf.keras.layers.Conv1D(
        model_width,
        kernel,
        strides=1,
        dilation_rate=dilation_rate,
        padding=padding,
        kernel_initializer="he_normal"
    )(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('selu')(x)
    return x


def Conv_1D_Block_APS(x, model_width, kernel, strides=1, padding="same"):
    """Convolution block with APS"""
    x = tf.keras.layers.Conv1D(
        model_width, kernel, strides=1,
        padding=padding, kernel_initializer="he_normal"
    )(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('selu')(x)

    if strides > 1:
        x = tf.keras.layers.Lambda(
            lambda t: APS_downsample_1D(t, stride=strides)
        )(x)

    return x


# ==================== Inception模块 ====================
def Inception_Module_A(inputs, filterB1_1, filterB2_1, filterB2_2,
                       filterB3_1, filterB3_2, filterB3_3, filterB4_1, i):
    """Standard Inception-A module"""
    branch1x1 = Conv_1D_Block(inputs, filterB1_1, 1)

    branch3x3 = Conv_1D_Block(inputs, filterB2_1, 1)
    branch3x3 = Conv_1D_Block(branch3x3, filterB2_2, 5)

    branch3x3dbl = Conv_1D_Block(inputs, filterB3_1, 1)
    branch3x3dbl = Conv_1D_Block(branch3x3dbl, filterB3_2, 3)
    branch3x3dbl = Conv_1D_Block(branch3x3dbl, filterB3_3, 3)

    branch_pool = tf.keras.layers.AveragePooling1D(
        pool_size=3, strides=1, padding='same'
    )(inputs)
    branch_pool = Conv_1D_Block(branch_pool, filterB4_1, 1)

    out = tf.keras.layers.concatenate(
        [branch1x1, branch3x3, branch3x3dbl, branch_pool],
        axis=-1,
        name='Inception_Block_A' + str(i)
    )

    return out


def Inception_Module_B(inputs, filterB1_1, filterB2_1, filterB2_2,
                       filterB3_1, filterB3_2, filterB3_3, filterB4_1, i):
    """Standard Inception-B module"""
    branch1x1 = Conv_1D_Block(inputs, filterB1_1, 1)

    branch3x3 = Conv_1D_Block(inputs, filterB2_1, 1)
    branch3x3 = Conv_1D_Block(branch3x3, filterB2_2, 7)
    branch3x3 = Conv_1D_Block(branch3x3, filterB2_1, 1)

    branch3x3dbl = Conv_1D_Block(inputs, filterB3_1, 1)
    branch3x3dbl = Conv_1D_Block(branch3x3dbl, filterB3_2, 7)
    branch3x3dbl = Conv_1D_Block(branch3x3dbl, filterB3_3, 1)
    branch3x3dbl = Conv_1D_Block(branch3x3dbl, filterB3_2, 7)
    branch3x3dbl = Conv_1D_Block(branch3x3dbl, filterB3_3, 1)

    branch_pool = tf.keras.layers.AveragePooling1D(
        pool_size=3, strides=1, padding='same'
    )(inputs)
    branch_pool = Conv_1D_Block(branch_pool, filterB4_1, 1)

    out = tf.keras.layers.concatenate(
        [branch1x1, branch3x3, branch3x3dbl, branch_pool],
        axis=-1,
        name='Inception_Block_B' + str(i)
    )

    return out


def Inception_Module_C(inputs, filterB1_1, filterB2_1, filterB2_2,
                       filterB3_1, filterB3_2, filterB3_3, filterB4_1, i):
    """Standard Inception-C module"""
    branch1x1 = Conv_1D_Block(inputs, filterB1_1, 1)

    branch3x3 = Conv_1D_Block(inputs, filterB2_1, 1)
    branch3x3left = Conv_1D_Block(branch3x3, filterB2_2, 3)
    branch3x3right = Conv_1D_Block(branch3x3, filterB2_2, 1, strides=1)
    convleftright1 = tf.concat([branch3x3left, branch3x3right], axis=2)

    branch3x3dbl1 = Conv_1D_Block(inputs, filterB3_1, 1)
    branch3x3dbl = Conv_1D_Block(branch3x3dbl1, filterB3_2, 3, strides=1)
    branch3x3dblleft = Conv_1D_Block(branch3x3dbl, filterB3_3, 3)
    branch3x3dblright = Conv_1D_Block(branch3x3dbl, filterB2_2, 1, strides=1)
    convleftright2 = tf.concat([branch3x3dblleft, branch3x3dblright], axis=2)

    branch_pool = tf.keras.layers.AveragePooling1D(
        pool_size=3, strides=1, padding='same'
    )(inputs)
    branch_pool = Conv_1D_Block(branch_pool, filterB4_1, 1)

    out = tf.keras.layers.concatenate(
        [branch1x1, convleftright1, convleftright2, branch_pool],
        axis=2,
        name='Inception_Block_C' + str(i)
    )

    return out


# ==================== Reduction模块(带APS)====================
def Reduction_Block_A_APS(inputs, filterB1_1, filterB1_2, filterB2_1, filterB2_2, filterB2_3, i):
    """Reduction Block A with APS"""
    branch3x3 = Conv_1D_Block(inputs, filterB1_1, 1)
    branch3x3 = Conv_1D_Block_APS(branch3x3, filterB1_2, 3, strides=2)

    branch3x3dbl = Conv_1D_Block(inputs, filterB2_1, 1)
    branch3x3dbl = Conv_1D_Block(branch3x3dbl, filterB2_2, 3)
    branch3x3dbl = Conv_1D_Block_APS(branch3x3dbl, filterB2_3, 3, strides=2)

    branch_pool = tf.keras.layers.Lambda(
        lambda x: APS_downsample_1D(x, stride=2)
    )(inputs)

    out = tf.keras.layers.concatenate(
        [branch3x3, branch3x3dbl, branch_pool],
        axis=-1,
        name='Reduction_Block_APS_' + str(i)
    )
    return out

# Reduction module (with APS)
def Reduction_Block_B_APS(inputs, filterB1_1, filterB1_2, filterB2_1, filterB2_2, filterB2_3, i):
    """Reduction Block B with APS"""
    branch3x3 = Conv_1D_Block(inputs, filterB1_1, 1)
    branch3x3 = Conv_1D_Block_APS(branch3x3, filterB1_2, 3, strides=2)

    branch3x3dbl = Conv_1D_Block(inputs, filterB2_1, 1)
    branch3x3dbl = Conv_1D_Block(branch3x3dbl, filterB2_2, 7)
    branch3x3dbl = Conv_1D_Block(branch3x3dbl, filterB2_1, 1)
    branch3x3dbl = Conv_1D_Block_APS(branch3x3dbl, filterB2_3, 3, strides=2)

    branch_pool = tf.keras.layers.Lambda(
        lambda x: APS_downsample_1D(x, stride=2)
    )(inputs)

    out = tf.keras.layers.concatenate(
        [branch3x3, branch3x3dbl, branch_pool],
        axis=-1,
        name='Reduction_Block_APS_' + str(i)
    )
    return out


output_nums = 256  # AES S-box output class count


# ==================== Main network: receptive field (RF=79) ====================
def Inceptionv3_APS_StandardReceptive_RF79(num_filters=64):

    input_shape = (20000, 1)
    inputs = tf.keras.Input(shape=input_shape)

    x = Conv_1D_Block(inputs, num_filters, 21, strides=1, padding='same')
    x = Conv_1D_Block(x, num_filters, 19, strides=1, padding='same')
    x = Conv_1D_Block_Dilated(x, num_filters, 17, dilation_rate=2, padding='same')
    x = Conv_1D_Block(x, num_filters * 2, 9, strides=1, padding='same')


    x = Conv_1D_Block(x, num_filters * 2, 11, strides=1, padding='valid')
    x = Conv_1D_Block_APS(x, num_filters * 2, 7, strides=2, padding='valid')
    x = Conv_1D_Block(x, num_filters * 2, 5, strides=1, padding='valid')
    x = tf.keras.layers.Lambda(lambda t: APS_downsample_1D(t, stride=2))(x)
    x = Conv_1D_Block(x, int(num_filters * 2.5), 7, strides=1, padding='valid')
    x = Conv_1D_Block_APS(x, num_filters * 6, 3, strides=2, padding='valid')


    # 2× Inception-A
    print("\n Building 2× Inception-A blocks...")
    x = Inception_Module_A(x, 64, 48, 64, 64, 96, 96, 32, 1)
    x = Inception_Module_A(x, 64, 48, 64, 64, 96, 96, 64, 2)
    x = tf.keras.layers.Lambda(lambda t: APS_downsample_1D(t, stride=2))(x)
    # Reduction Block A
    print("\n Building Reduction-A block (with APS)...")
    x = Reduction_Block_A_APS(x, 64, 384, 64, 96, 96, 1)
    # 3× Inception-B
    print("\n Building 3× Inception-B blocks...")
    x = Inception_Module_B(x, 192, 128, 192, 128, 128, 192, 192, 1)
    x = Inception_Module_B(x, 192, 160, 192, 160, 160, 192, 192, 2)
    x = Inception_Module_B(x, 192, 160, 192, 160, 160, 192, 192, 3)
    x = tf.keras.layers.Lambda(lambda t: APS_downsample_1D(t, stride=2))(x)
    # Reduction Block B
    print("\n Building Reduction-B block (with APS)...")
    x = Reduction_Block_B_APS(x, 192, 320, 192, 192, 192, 2)
    # 1× Inception-C
    print("\n Building 1× Inception-C block...")
    x = Inception_Module_C(x, 320, 384, 384, 448, 384, 384, 192, 1)


    # Classification
    print("\n Building classification head...")
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = tf.keras.layers.Flatten(name='flatten')(x)
    x = tf.keras.layers.Dense(4096, activation='relu', name='fc1')(x)
    x = tf.keras.layers.Dropout(0.3, name='dropout1')(x)
    x = tf.keras.layers.Dense(4096, activation='relu', name='fc2')(x)
    x = tf.keras.layers.Dropout(0.4, name='dropout2')(x)

    outputs = tf.keras.layers.Dense(output_nums, activation='softmax', name='output')(x)
    model = tf.keras.Model(inputs, outputs, name='InceptionNet_APS_LargeReceptive_RF79')
    optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.0001)
    model.compile(
        loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
        optimizer=optimizer,
        metrics=['accuracy']
    )
    return model



def train_model(X_profiling, Y_profiling, model, save_file_name,
                X_validation=None, Y_validation=None,
                epochs=100, batch_size=200, history_file=None):
    """
    Train model (supports validation set)

    Args:
        X_profiling: Training set features
        Y_profiling: Training set labels
        model: Model
        save_file_name: Model save path
        X_validation: Validation set features (optional)
        Y_validation: Validation set labels (optional)
        epochs: Number of training epochs
        batch_size: Batch size
        history_file: Training history save path
    """
    check_file_exists(os.path.dirname(save_file_name))

    # Reshape data into CNN format
    Reshaped_X_profiling = X_profiling.reshape((X_profiling.shape[0], X_profiling.shape[1], 1))
    Y_profiling_cat = tf.keras.utils.to_categorical(Y_profiling, num_classes=256)
    print(f"\n Training data shape: {Reshaped_X_profiling.shape}")

    # Prepare validation data
    validation_data = None
    if X_validation is not None and Y_validation is not None:
        Reshaped_X_validation = X_validation.reshape((X_validation.shape[0], X_validation.shape[1], 1))
        Y_validation_cat = tf.keras.utils.to_categorical(Y_validation, num_classes=256)
        validation_data = (Reshaped_X_validation, Y_validation_cat)
        print(f" Validation data shape: {Reshaped_X_validation.shape}")


    print("\n️ Setting up callbacks...")
    callbacks = []

    if X_validation is not None:

        best_model_path = save_file_name.replace('.h5', '_best.h5')
        callbacks.append(
            tf.keras.callbacks.ModelCheckpoint(
                best_model_path,
                monitor='val_accuracy',
                save_best_only=True,
                mode='max',
                verbose=1
            )
        )
        print(f"     ModelCheckpoint: save BEST model to {os.path.basename(best_model_path)}")
        print(f"     (Monitor metric: val_accuracy, Save strategy: best only)")


        checkpoint_dir = os.path.dirname(save_file_name)
        checkpoint_path = os.path.join(checkpoint_dir, 'checkpoints',
                                       os.path.basename(save_file_name).replace('.h5', '_epoch{epoch:03d}.h5'))
        os.makedirs(os.path.join(checkpoint_dir, 'checkpoints'), exist_ok=True)

        callbacks.append(
            tf.keras.callbacks.ModelCheckpoint(
                checkpoint_path,
                monitor='val_accuracy',
                save_best_only=False,
                save_freq='epoch',
                period=5,  # Save every 5 epochs
                verbose=1
            )
        )
        print(f"   ✓ ModelCheckpoint: save model every 5 epochs")
        print(f"     (Save location: checkpoints/ folder)")

    else:

        callbacks.append(
            tf.keras.callbacks.ModelCheckpoint(
                save_file_name,
                monitor='loss',
                save_best_only=True,
                mode='min',
                verbose=1
            )
        )
        print("    ModelCheckpoint: save best model based on training loss")
        print("    No validation set - monitoring training metrics only")


    print("\n" + "=" * 80)
    print("MODEL SUMMARY")
    print("=" * 80)
    model.summary()
    print("=" * 80 + "\n")


    print("\n" + "=" * 80)
    print("STARTING TRAINING")
    print("=" * 80)
    print(f"Epochs: {epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Fixed learning rate: 0.0001")
    if validation_data is not None:
        print(f"Validation:  Enabled ({X_validation.shape[0]} samples)")
    else:
        print(f"Validation:  Disabled")
    print("=" * 80 + "\n")

    history = model.fit(
        x=Reshaped_X_profiling,
        y=Y_profiling_cat,
        batch_size=batch_size,
        epochs=epochs,
        verbose=1,
        callbacks=callbacks,
        validation_data=validation_data
    )


    if history_file is not None:
        with open(history_file, "wb") as f:
            pickle.dump(history.history, f)
        print(f"\n💾 Training history saved to: {history_file}")

    return history


def find_latest_checkpoint(model_folder, model_name_base):
    """Find the latest checkpoint file"""

    base_name = model_name_base.replace(".h5", "")
    checkpoint_pattern = os.path.join(model_folder, "checkpoints", base_name + "_epoch*.h5")
    checkpoints = glob.glob(checkpoint_pattern)

    if not checkpoints:
        return None, 0

    checkpoint_epochs = []
    for ckpt in checkpoints:
        try:
            epoch_str = ckpt.rsplit("_epoch", 1)[1].split(".h5")[0]
            epoch_num = int(epoch_str)
            checkpoint_epochs.append((ckpt, epoch_num))
        except:
            continue

    if not checkpoint_epochs:
        return None, 0

    checkpoint_epochs.sort(key=lambda x: x[1])
    latest_ckpt, latest_epoch = checkpoint_epochs[-1]

    print(f"\n📂 Found {len(checkpoint_epochs)} checkpoint(s) for this model:")
    for ckpt, ep in checkpoint_epochs[-3:]:
        size_mb = os.path.getsize(ckpt) / (1024 * 1024)
        print(f"   • Epoch {ep:3d}: {os.path.basename(ckpt)} ({size_mb:.1f} MB)")

    return latest_ckpt, latest_epoch



if __name__ == "__main__":
    # Dataset file path
    ASCAD_data_folder = "./ASCAD/ASCAD_dataset/"
    #trained_models_dir
    ASCAD_trained_models_folder = "./ASCAD/ASCAD_trained_models/"
    # Directory for saving training history files
    history_folder = "./ASCAD/training_history/"


    os.makedirs(ASCAD_trained_models_folder, exist_ok=True)
    os.makedirs(history_folder, exist_ok=True)
    os.makedirs(os.path.join(ASCAD_trained_models_folder, 'checkpoints'), exist_ok=True)

    start = time.time()

    # ==================== Load data (excluding attack data) ====================
    print(" Loading ASCADfL Desync0 dataset (TRAINING MODE)...")
    (X_profiling, Y_profiling), (X_validation, Y_validation), _ = load_ascad(
        ASCAD_data_folder + "ATMega8515_raw_traces_20k_desync0.h5",
        load_metadata=False,
        validation_split=5000,
        load_attack=False
    )

    # ==================== Training parameters ====================
    epochs = 400
    batch_size = 50

    # ==================== Create model ====================

    #  Check for checkpoint
    model_name = f"InceptionNet_APS_ascadfL_desync0_20k_epochs{epochs}_batchsize{batch_size}.h5"

    latest_checkpoint, completed_epoch = find_latest_checkpoint(
        ASCAD_trained_models_folder,
        model_name
    )


    initial_epoch = 0  # 默认从0开始

    if latest_checkpoint is not None and completed_epoch < epochs:

        print("\n" + "=" * 80)
        print(" RESUMING FROM CHECKPOINT")
        print("=" * 80)
        print(f" Found checkpoint at epoch {completed_epoch}")
        print(f" File: {os.path.basename(latest_checkpoint)}")
        print(f" Will resume training from epoch {completed_epoch} to {epochs}")
        print(f" Remaining epochs: {epochs - completed_epoch}")
        print("=" * 80 + "\n")

        try:
            model = tf.keras.models.load_model(latest_checkpoint)
            initial_epoch = completed_epoch
            print(f"✅ Model loaded successfully!")
            print(f"   Total parameters: {model.count_params():,}")
        except Exception as e:
            print(f" Error loading checkpoint: {e}")
            print("️  Will create new model instead...")
            model = Inceptionv3_APS_StandardReceptive_RF79(num_filters=64)
            initial_epoch = 0

    elif latest_checkpoint is not None and completed_epoch >= epochs:

        print("\n" + "=" * 80)
        print("✅ TRAINING ALREADY COMPLETED")
        print("=" * 80)
        print(f" Found checkpoint at epoch {completed_epoch}/{epochs}")
        print(f" Training target ({epochs} epochs) already reached!")
        print(f" Model: {os.path.basename(latest_checkpoint)}")
        print("=" * 80 + "\n")
        sys.exit(0)

    else:

        print("\n" + "=" * 80)
        print(" BUILDING NEW MODEL (no checkpoint found)")
        print("=" * 80)
        print(" Starting fresh training from epoch 0")
        print("=" * 80 + "\n")

        model = Inceptionv3_APS_StandardReceptive_RF79(num_filters=64)
        initial_epoch = 0

    print(f"\n Model ready: {model_name}")
    print(f"   Total parameters: {model.count_params():,}")
    print(f"   Training range: epoch {initial_epoch} → {epochs}")



    history_file = os.path.join(
        history_folder,
        "history_" + model_name.replace(".h5", ".pkl")
    )

    print("\n Creating memory-efficient generator...")

    import gc

    gc.collect()


    def data_generator(X, Y, batch_size, shuffle=True):

        indices = np.arange(len(X))

        while True:
            if shuffle:
                np.random.shuffle(indices)

            for start_idx in range(0, len(X), batch_size):
                end_idx = min(start_idx + batch_size, len(X))
                batch_indices = indices[start_idx:end_idx]

                batch_x = X[batch_indices].reshape(-1, 20000, 1)
                batch_y = tf.keras.utils.to_categorical(Y[batch_indices], num_classes=256)

                yield batch_x, batch_y



    steps_per_epoch = int(np.ceil(len(X_profiling) / batch_size))
    validation_steps = int(np.ceil(len(X_validation) / batch_size))

    print(f"   Generator created (memory efficient!)")
    print(f"   Training steps per epoch: {steps_per_epoch}")
    print(f"   Validation steps: {validation_steps}")
    print(f"   Memory per batch: ~{batch_size * 20000 * 8 / 1024 ** 2:.1f} MB")

    # Set up callbacks
    print("\n️ Setting up callbacks...")
    callbacks = []

    # Callback 1: Save best model
    best_model_path = ASCAD_trained_models_folder + model_name.replace('.h5', '_best.h5')
    callbacks.append(
        tf.keras.callbacks.ModelCheckpoint(
            best_model_path,
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        )
    )
    print(f"   ✓ ModelCheckpoint: save BEST model to {os.path.basename(best_model_path)}")

    # Callback 2: Save checkpoint every 5 epochs
    checkpoint_path = os.path.join(
        ASCAD_trained_models_folder, 'checkpoints',
        model_name.replace('.h5', '_epoch{epoch:03d}.h5')
    )
    callbacks.append(
        tf.keras.callbacks.ModelCheckpoint(
            checkpoint_path,
            monitor='val_accuracy',
            save_best_only=False,
            save_freq='epoch',
            period=5,
            verbose=1
        )
    )
    print(f"    ModelCheckpoint: save every 5 epochs to checkpoints/")


    if initial_epoch == 0:
        print("\n" + "=" * 80)
        print("MODEL SUMMARY")
        print("=" * 80)
        model.summary()
        print("=" * 80 + "\n")


    print("\n" + "=" * 80)
    if initial_epoch > 0:
        print(" RESUMING TRAINING")
    else:
        print(" STARTING TRAINING")
    print("=" * 80)
    print(f"Start epoch: {initial_epoch}")
    print(f"End epoch: {epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {model.optimizer.learning_rate.numpy():.6f}")
    print(f"Validation:  Enabled ({X_validation.shape[0]} samples)")
    print(f"Training data size: {X_profiling.shape[0]} samples ({X_profiling.nbytes / 1024 ** 3:.2f} GB)")
    print("=" * 80 + "\n")


    history = model.fit(
        data_generator(X_profiling, Y_profiling, batch_size, shuffle=True),
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        initial_epoch=initial_epoch,
        verbose=1,
        callbacks=callbacks,
        validation_data=data_generator(X_validation, Y_validation, batch_size, shuffle=False),
        validation_steps=validation_steps
    )


    with open(history_file, "wb") as f:
        pickle.dump(history.history, f)
    print(f"\n Training history saved to: {history_file}")


    print("\n" + "=" * 80)
    print("💾 SAVING MODELS")
    print("=" * 80)

    # Save 1: Final epoch model (final)
    final_save_path = ASCAD_trained_models_folder + model_name.replace('.h5', '_final.h5')
    print(f"\n  Saving FINAL model (last epoch): {os.path.basename(final_save_path)}")

    try:
        model.save(final_save_path)

        if os.path.exists(final_save_path):
            size_mb = os.path.getsize(final_save_path) / (1024 * 1024)
            print(f"    FINAL model saved successfully!")
            print(f"    File size: {size_mb:.2f} MB")
            print(f"    Path: {final_save_path}")
        else:
            print("     Warning: FINAL model file not found after save!")

    except Exception as e:
        print(f"   ❌ Error saving FINAL model: {e}")
        import traceback

        traceback.print_exc()

    # Save 2: Best validation accuracy model (best) - already auto-saved by ModelCheckpoint
    best_save_path = ASCAD_trained_models_folder + model_name.replace('.h5', '_best.h5')
    print(f"\n BEST model (highest val_accuracy): {os.path.basename(best_save_path)}")

    if os.path.exists(best_save_path):
        size_mb = os.path.getsize(best_save_path) / (1024 * 1024)
        print(f"    BEST model already saved by ModelCheckpoint!")
        print(f"    File size: {size_mb:.2f} MB")
        print(f"    Path: {best_save_path}")
    else:
        print(f"   ️Warning: BEST model not found at {best_save_path}")
        print(f"    Tip: This happens if training didn't complete any epochs")

    print("\n" + "=" * 80)
    print(" Model Saving Summary:")
    print("=" * 80)
    print(f"   • BEST model (best val_accuracy):  {os.path.basename(best_save_path)}")
    print(f"   • FINAL model (last epoch):         {os.path.basename(final_save_path)}")
    print("=" * 80)

    # ==================== Training Complete ====================
    end = time.time()
    elapsed = end - start

    print("\n" + "=" * 80)
    print(" TRAINING COMPLETED")
    print("=" * 80)
    print(f"️  Total time: {elapsed:.0f} seconds ({elapsed / 60:.1f} minutes)")
    print(f" BEST Model: {best_save_path}")
    print(f" FINAL Model: {final_save_path}")
    print(f" History: {history_file}")
    print("=" * 80 + "\n")

    #  Display final training results (including validation set metrics)
    print(" Final Training Results:")
    final_train_acc = history.history['accuracy'][-1]
    final_train_loss = history.history['loss'][-1]
    best_train_loss = min(history.history['loss'])

    print(f"   Training accuracy: {final_train_acc * 100:.2f}%")
    print(f"   Training loss: {final_train_loss:.4f}")
    print(f"   Best training loss: {best_train_loss:.4f}")


    if 'val_accuracy' in history.history:
        final_val_acc = history.history['val_accuracy'][-1]
        final_val_loss = history.history['val_loss'][-1]
        best_val_acc = max(history.history['val_accuracy'])
        best_val_loss = min(history.history['val_loss'])

        print(f"\n   Validation accuracy: {final_val_acc * 100:.2f}%")
        print(f"   Validation loss: {final_val_loss:.4f}")
        print(f"   Best validation accuracy: {best_val_acc * 100:.2f}%")
        print(f"   Best validation loss: {best_val_loss:.4f}")

        best_val_epoch = np.argmax(history.history['val_accuracy']) + 1
        print(f"   Best validation epoch: {best_val_epoch}/{len(history.history['val_accuracy'])}")



    best_train_epoch = np.argmin(history.history['loss']) + 1
    print(f"\n   Best training epoch: {best_train_epoch}/{len(history.history['loss'])}")

    print("=" * 80 + "\n")