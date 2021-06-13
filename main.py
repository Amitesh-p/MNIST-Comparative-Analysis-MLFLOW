import tensorflow as tf
import mlflow
import mlflow.keras
from tensorflow.keras.utils import to_categorical

fashion = tf.keras.datasets.fashion_mnist
(X_train, Y_train), (X_test, Y_test) = fashion.load_data()

X_train = X_train.reshape((60000, 784)).astype("float32")
X_test = X_test.reshape((10000, 784)).astype("float32")

X_train = X_train / 255.
X_test = X_test / 255.

Y_train = to_categorical(Y_train)
Y_test = to_categorical(Y_test)


def runModel(activation):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Flatten(input_shape=(784,), name="inputLayer"))
    model.add(tf.keras.layers.Dense(200, activation=activation, name="hiddenLayer1"))
    model.add(tf.keras.layers.Dense(200, activation=activation, name="hiddenLayer2"))
    model.add(tf.keras.layers.Dense(100, activation=activation, name="hiddenLayer3"))
    model.add(tf.keras.layers.Dense(10, activation='softmax', name="outputLayer"))

    with mlflow.start_run() as run:
        model.compile(loss="categorical_crossentropy", optimizer='Adam', metrics=['accuracy'])
        model.fit(X_train, Y_train, batch_size=32, epochs=10, verbose=0, validation_data=(X_test, Y_test))
        # Evaluate our model
        score = model.evaluate(X_test, Y_test, verbose=0)

        # Log Parameters
        mlflow.log_param("activation function", activation)
        mlflow.log_metric("test loss", score[0])
        mlflow.log_metric("test accuracy", score[1])

        # Log Model
        mlflow.keras.log_model(model, "model")
        return score


score_sigmoid = runModel('tanh')
print('Test loss:', score_sigmoid[0])
print('Test accuracy:', score_sigmoid[1])