from openai import OpenAI

class DeepSeek():
    
    def __init__(self):
        self.client = OpenAI(api_key="sk-f3d78103df6145e6a20a442fcdc891b0", base_url="https://api.deepseek.com")

    def __call__(self, sys_prompt, usr_prompt, **kwargs):
        response = self.client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": usr_prompt},
            ],
            stream=False
        )

        answer = response.choices[0].message.content
        return answer
