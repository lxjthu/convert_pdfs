from openai import OpenAI
import os
import requests
import time
import urllib3

def test_qwen_connection():
    try:
        # 检查API密钥
        api_key = os.environ.get('DASHSCOPE_API_KEY')
        if not api_key:
            print("错误：未找到 DASHSCOPE_API_KEY 环境变量")
            return False

        # 禁用 SSL 警告（仅用于测试）
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        
        # 创建客户端（使用兼容模式的API地址）
        client = OpenAI(
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            api_key=api_key,
            timeout=30
        )
        
        # 测试网络连接
        try:
            response = requests.post(
                "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions", 
                timeout=5,
                verify=False,
                headers={
                    'Authorization': f'Bearer {api_key}',
                    'Content-Type': 'application/json'
                },
                json={
                    "model": "qwen-turbo",
                    "messages": [
                        {"role": "user", "content": "测试消息"}
                    ]
                }
            )
            print(f"API 状态码: {response.status_code}")
            print(f"响应内容: {response.text}")
        except requests.exceptions.RequestException as e:
            print(f"网络连接测试失败: {str(e)}")
            return False

        # 尝试3次请求
        for attempt in range(3):
            try:
                print(f"尝试第 {attempt + 1} 次连接...")
                response = client.chat.completions.create(
                    model="qwen-turbo",
                    messages=[
                        {'role': 'system', 'content': '你是一个助手'},
                        {'role': 'user', 'content': '你好，这是一个测试消息。'}
                    ]
                )
                print("连接成功！")
                print("模型响应:", response.choices[0].message.content)
                return True
            except Exception as e:
                print(f"第 {attempt + 1} 次尝试失败: {str(e)}")
                if attempt < 2:
                    print("等待3秒后重试...")
                    time.sleep(3)
        
        return False
        
    except Exception as e:
        print("连接失败！")
        print("错误信息:", str(e))
        return False

if __name__ == "__main__":
    print("开始测试通义千问API连接...")
    test_qwen_connection()