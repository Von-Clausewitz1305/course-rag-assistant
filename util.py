from llama_index.core import SimpleDirectoryReader
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core.schema import Document
from typing import List, Generator

from llama_index.llms.dashscope import DashScope, DashScopeGenerationModels

import config


# 把目录下面所有的文件都解析出来
def parse_all_formats(input_dir):

    reader = SimpleDirectoryReader(
        input_dir = input_dir
    )

    return reader.load_data()

# 把目录下面所有的问文件清洗一下
def clean_all_formats(input_dir) -> List[Document]:
    import dataClean

    docs = parse_all_formats(input_dir)
    # 清洗之后的对象数组
    cleaned_docs = []

    for i, doc in enumerate(docs):
        cleand_doc = dataClean.clean_doc(doc)
        cleaned_docs.append(cleand_doc)

    return cleaned_docs

# 专门初始化一下通义千问的客户端实例
def create_qwen_turbo_llm (
    temperature: float=0.5,
    max_tokens: int=1024,
) -> DashScope:

    # 1. 获取key
    key = config.dashscope_api_key

    # 2. 创建客户端实例
    return DashScope(
        model_name=DashScopeGenerationModels.QWEN_TURBO,
        api_key=key,
        temperature=temperature,
        max_tokens=max_tokens,
    )

# 创建LLM实例
llm = create_qwen_turbo_llm()

# 单独对话
def complete(
        prompt: str
) -> str:
    return llm.complete(prompt=prompt).text


# 单独对话（流式返回）
def stream_complete(
        prompt: str
) -> Generator[str, None, None]:
    for chunk in llm.stream_complete(prompt=prompt):
        if chunk.delta:
            yield chunk.delta


# 多轮对话
def chat(
        user_prompt: str,
        sys_prompt: str,
) -> str:
    messages = [
        ChatMessage(role = MessageRole.USER, content=user_prompt),
        ChatMessage(role = MessageRole.SYSTEM, content=sys_prompt),
    ]
    return llm.chat(messages).message.content

# 多轮对话（流式）
def stream_chat(
    user_prompt: str,
    sys_prompt: str,
) -> Generator[str, None, None]:
    messages = [
        ChatMessage(role=MessageRole.USER, content=user_prompt),
        ChatMessage(role=MessageRole.SYSTEM, content=sys_prompt),
    ]
    for chunk in llm.stream_chat(messages):
        if chunk.delta:
            yield chunk.delta



if __name__ == "__main__":
    # print(complete("介绍一下什么是RAG，包括它的起源、发展和现状"))
    # for text in stream_complete("介绍一下什么是RAG，包括它的起源、发展和现状"):
    #     print(text, end="", flush=True)
    # print(chat("陕西的省会是哪个城市？", "只回答城市名即可"))
    for text in stream_chat("介绍一下什么是RAG，包括它的起源、发展和现状", "限制在300字以内"):
        print(text, end="", flush=True)
    print()