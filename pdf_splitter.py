import os
import fitz
import dashscope
from dashscope import Generation
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
import spacy
import re

class PDFSplitter:
    def __init__(self, api_key):
        self.api_key = api_key
        dashscope.api_key = api_key
        # 加载中文NLP模型
        self.nlp = spacy.load("zh_core_web_sm")
        
    def initial_split(self, text):
        """使用LangChain进行初步分块"""
        # 使用RecursiveCharacterTextSplitter进行初步分块
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", "。", "，", " "]
        )
        
        # 初步分块
        chunks = text_splitter.split_text(text)
        
        # 使用spaCy进行语义分析
        processed_chunks = []
        for chunk in chunks:
            doc = self.nlp(chunk)
            # 判断是否是完整的语义单元
            if doc.has_annotation("SENT_START"):
                processed_chunks.append(chunk)
        
        return processed_chunks
    
    def refine_chunks(self, chunks):
        """使用通义千问优化分块"""
        messages = [
            {
                'role': 'user',
                'content': f'''请分析以下文本块，判断是否需要合并或分割，并给出优化建议：
                {chunks[:3]}  # 每次处理前3个块
                '''
            }
        ]
        
        response = Generation.call(
            model='qwen-max',
            messages=messages
        )
        
        return self._process_refinement_response(response.output.text, chunks)
    
    def _process_refinement_response(self, response, chunks):
        """处理大模型的优化建议"""
        # 这里添加处理逻辑
        return refined_chunks
    
    def create_markdown_files(self, chunks, output_dir):
        """根据分块创建Markdown文件"""
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        for i, chunk in enumerate(chunks):
            # 使用大模型生成标题
            title = self._generate_title(chunk)
            file_name = f"{i+1}-{title}.md"
            file_path = os.path.join(output_dir, file_name)
            
            # 创建文件内容
            content = self._create_markdown_content(i, title, chunk, len(chunks))
            
            # 写入文件
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
    
    def _generate_title(self, chunk):
        """使用通义千问生成适当的标题"""
        messages = [
            {
                'role': 'user',
                'content': f'请为以下内容生成一个简短的标题：\n{chunk[:200]}'
            }
        ]
        response = Generation.call(model='qwen-max', messages=messages)
        return response.output.text.strip()

def main():
    api_key = "your_api_key_here"
    splitter = PDFSplitter(api_key)
    
    pdf_path = r"E:\pdf文档转换\input.pdf"
    output_dir = r"E:\pdf文档转换\markdown_files"
    
    # 1. 提取文本
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    
    # 2. 初步分块
    initial_chunks = splitter.initial_split(text)
    
    # 3. 优化分块
    refined_chunks = splitter.refine_chunks(initial_chunks)
    
    # 4. 生成Markdown文件
    splitter.create_markdown_files(refined_chunks, output_dir)

if __name__ == "__main__":
    main()