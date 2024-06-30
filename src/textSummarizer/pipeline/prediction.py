from textSummarizer.config.configuration import ConfigurationManager
from transformers import AutoTokenizer, pipeline

class PredictionPipeline:
    def __init__(self):
        self.config = ConfigurationManager().get_model_evaluation_config()

    def predict(self, text):
        tokenizer = AutoTokenizer.from_pretrained(self.config.tokenizer_path)
        gen_kwargs = {"length_penalty": 0.8, "num_beams": 8, "max_length": 56}

        # Initialize the pipeline
        pipe = pipeline("text2text-generation", model=self.config.model_path, tokenizer=tokenizer)

        print("Dialogue:")
        print(text)

        try:
            output = pipe(text, **gen_kwargs)
            print("Pipeline output:", output)  # Debugging line to check output structure

            if output and "generated_text" in output[0]:
                summary = output[0]["generated_text"]
                print("\nModel Summary:")
                print(summary)
                return summary
            else:
                raise KeyError("The key 'generated_text' was not found in the output.")
        except Exception as e:
            print(f"An error occurred during prediction: {e}")
            return None
