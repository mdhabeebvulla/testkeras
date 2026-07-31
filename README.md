# Iris Keras Gradio App

## Deploy on Render

1. Upload all files in this folder to a GitHub repository.
2. In Render, select **New > Web Service** and connect the repository.
3. Use:
   - Build command: `pip install -r requirements.txt`
   - Start command: `python app.py`
4. Select the Free instance and deploy.

The app binds to `0.0.0.0` and uses Render's `PORT` environment variable.

## Local run

```bash
pip install -r requirements.txt
python app.py
```

## Preprocessing note

The original scaler file was not supplied. `app.py` uses the mean and scale from the full sklearn Iris dataset. For exact agreement with training, replace those constants with the values from the original fitted `StandardScaler`, or save and load that scaler separately.
