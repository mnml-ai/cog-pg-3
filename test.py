class MLSDdetector:
    @classmethod
    def from_pretrained(cls, model_name):
        # Implementation of the from_pretrained method
        # Replace this with the actual implementation

        # For demonstration purposes, let's print a message
        print(f"Initializing MLSDdetector with model: {model_name}")

        # Return an instance of the class
        return cls()

# Initialize AUX_IDS with a lambda function
AUX_IDS = {
    "scribble": {
        "path": "fusing/stable-diffusion-v1-5-controlnet-scribble",
        "detector": lambda: MLSDdetector.from_pretrained("lllyasviel/ControlNet")
    }
}

# Later in your code, when you want to use the detector:
# Call the lambda function to get the initialized detector
# detector_instance = AUX_IDS["scribble"]["detector"]()