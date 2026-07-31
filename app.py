import os
import gradio as gr
import numpy as np
import keras

model = keras.models.load_model("iris_mlp.keras", compile=False)
CLASSES = ["setosa", "versicolor", "virginica"]

# StandardScaler values from the complete sklearn Iris dataset.
MEAN = np.array([5.8433332, 3.0573332, 3.7580000, 1.1993333], dtype=np.float32)
SCALE = np.array([0.8253013, 0.4344110, 1.7594041, 0.7596926], dtype=np.float32)


def predict(sepal_length, sepal_width, petal_length, petal_width):
    x = np.array(
        [[sepal_length, sepal_width, petal_length, petal_width]],
        dtype=np.float32,
    )
    x = (x - MEAN) / SCALE
    probabilities = model.predict(x, verbose=0)[0]
    return {name: float(probabilities[i]) for i, name in enumerate(CLASSES)}


demo = gr.Interface(
    fn=predict,
    inputs=[
        gr.Slider(4.0, 8.0, value=5.1, step=0.1, label="Sepal length (cm)"),
        gr.Slider(2.0, 4.5, value=3.5, step=0.1, label="Sepal width (cm)"),
        gr.Slider(1.0, 7.0, value=1.4, step=0.1, label="Petal length (cm)"),
        gr.Slider(0.1, 2.5, value=0.2, step=0.1, label="Petal width (cm)"),
    ],
    outputs=gr.Label(num_top_classes=3, label="Predicted species"),
    title="Iris Classifier - Keras 4-8-10-10-3",
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
