import litellm

class APIModel:
    def __init__(self, model="gpt-4"):
        self.model = model

    def complete(self, prompt, **kwargs):
        response = litellm.completion(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            **kwargs
        )
        return response.choices[0].message.content
