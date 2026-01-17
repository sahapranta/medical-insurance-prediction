import gradio as gr
import pandas as pd
import pickle

class MedicalInsurancePredictor:
    def __init__(self, model_path="grid_search_model.pkl"):
        self.model = self._load_model(model_path)
        self.feature_names = ['age','sex','bmi','children','smoker','region']

    def _load_model(self, model_path):        
        try:
            with open(model_path, "rb") as f:
                model = pickle.load(f)
            print(f"✓ Model loaded successfully from {model_path}")
            return model
        except Exception as e:
            print(f"✗ Error loading model: {e}")
            raise

    def predict(self, *args):
        try:
            if len(args) != len(self.feature_names):
                    return "Error: Invalid number of inputs"

            data = pd.DataFrame([args], columns=self.feature_names)

            data['bmi_category'] = pd.cut(
                data['bmi'],
                bins=[0, 18.5, 25, 30, 100],
                labels=["Underweight", "Normal", "Overweight", "Obese"]
            )

            prediction = self.model.predict(data)[0]
            return f"Estimated Insurance Cost: ${prediction:,.2f}"
                 
        except Exception as e:
            return f"Error making prediction: {str(e)}"

def create_interface():
    predictor = MedicalInsurancePredictor()

    inputs = [
        gr.Number(label="Age", value=18, minimum=12, maximum=65),
        gr.Radio(["male", "female"], label="Sex", value="male"),
        gr.Number(label="BMI", value=19, minimum=0, maximum=100),
        gr.Number(label="Children", value=1, minimum=0, maximum=10),
        gr.Radio(["Yes", "No"], label="Smoker", value="No"),
        gr.Dropdown(["southwest", "southeast", "northwest", "northeast"], label="Region")
    ]
    
    features = ", ".join(predictor.feature_names)
    interface = gr.Interface(
        fn=predictor.predict,
        inputs=inputs,
        outputs=gr.Textbox(label="Prediction Result"),
        title="Medical Insurance Cost Prediction",
        description=(
            "Predict Medical Insurance Cost based on:  " + 
            features
        ),
    )

    return interface


def main():
    app = create_interface()
    app.launch(share=True)

if __name__ == "__main__":
    main()
