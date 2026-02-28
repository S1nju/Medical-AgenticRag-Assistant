from langchain_qwq import ChatQwen
from app.core.config import settings
class ChatModelSinglton():
    model = None
    def get_model_instance(self, model_name: str, temperature: float = 0.7):
        if self.model is None:
            self.model = ChatQwen(model=model_name, temperature=temperature,api_key=settings.dashscope_api_key)
        return self.model