import os

import gradio as gr
import numpy as np
import tensorflow as tf

model = tf.keras.models.load_model("iris_mlp.keras")

CLASSES = ["setosa", "versicolor", "virginica"]

# StandardScaler values from the Iris dataset.
MEAN = np.array(
    [5.843333, 3.057333, 3.758000, 1.199333],
    dtype=np.float32,
)

SCALE = np.array(
    [0.825301, 0.434411, 1.759404, 0.759693],
    dtype=np.float32,
)


def predict(sepal_length, sepal_width, petal_length, petal_width):
    x = np.array(
        [[sepal_length, sepal_width, petal_length, petal_width]],
        dtype=np.float32,
    )

    x_scaled = (x - MEAN) / SCALE
    probabilities = model.predict(x_scaled, verbose=0)[0]

    return {
        CLASSES[i]: float(probabilities[i])
        for i in range(len(CLASSES))
    }


demo = gr.Interface(
    fn=predict,
    inputs=[
        gr.Slider(4.0, 8.0, value=5.1, step=0.1, label="Sepal length (cm)"),
        gr.Slider(2.0, 4.5, value=3.5, step=0.1, label="Sepal width (cm)"),
        gr.Slider(1.0, 7.0, value=1.4, step=0.1, label="Petal length (cm)"),
        gr.Slider(0.1, 2.5, value=0.2, step=0.1, label="Petal width (cm)"),
    ],
    outputs=gr.Label(
        num_top_classes=3,
        label="Predicted species",
    ),
    title="Iris Classifier",
    examples=[
        [5.1, 3.5, 1.4, 0.2],
        [6.0, 2.7, 4.2, 1.3],
        [6.9, 3.1, 5.4, 2.1],
    ],
    flagging_mode="never",
)

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 7860)),
    )
