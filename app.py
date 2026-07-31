import os

import gradio as gr
import numpy as np
import tensorflow as tf
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler


# Load the trained Keras model
model = tf.keras.models.load_model("iris_mlp.keras")

# Load Iris dataset to create the same StandardScaler
iris = load_iris()
CLASSES = list(iris.target_names)

scaler = StandardScaler()
scaler.fit(iris.data)


def predict(sepal_length, sepal_width, petal_length, petal_width):
    values = [
        sepal_length,
        sepal_width,
        petal_length,
        petal_width,
    ]

    if any(value is None for value in values):
        raise gr.Error("Please enter all four measurements.")

    x = np.array([values], dtype=np.float32)

    # Apply StandardScaler
    x_scaled = scaler.transform(x)

    # Make prediction
    probabilities = model.predict(x_scaled, verbose=0)[0]

    return {
        CLASSES[i]: float(probabilities[i])
        for i in range(len(CLASSES))
    }


demo = gr.Interface(
    fn=predict,
    inputs=[
        gr.Number(
            value=5.1,
            label="Sepal length (cm)",
        ),
        gr.Number(
            value=3.5,
            label="Sepal width (cm)",
        ),
        gr.Number(
            value=1.4,
            label="Petal length (cm)",
        ),
        gr.Number(
            value=0.2,
            label="Petal width (cm)",
        ),
    ],
    outputs=gr.Label(
        num_top_classes=3,
        label="Predicted species",
    ),
    title="Iris Classifier",
    description="Enter the four flower measurements and click Submit.",
    
)


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 7860)),
    )
