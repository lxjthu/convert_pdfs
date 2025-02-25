import os
import re
from pathlib import Path
from format_paper import PaperFormatter
from dashscope import Generation

class AIEnhancedFormatter:
    def __init__(self):
         # 只启用需要的基础格式化功能
        self.paper_formatter = PaperFormatter(features=[
            'title_format',  # 标题格式化
            'line_merge',    # 行合并
            'image_adjust',  # 图片位置调整
            'batch_process'   # 批处理功能
        ])
        self.api_key = os.environ.get('DASHSCOPE_API_KEY')
        if not self.api_key:
            raise ValueError("请在环境变量中设置 DASHSCOPE_API_KEY")
    
    def _enhance_with_ai(self, text):
        """使用通义千问增强文本内容"""
        messages = [
            {
                'role': 'user',
                'content': f'''请对以下学术文本进行整理和优化，要求：
                1. 调整顺序混乱的句子，使文章结构更加清晰
                2. 当句子出现明显断裂时，尝试将其合并为一个完整的句子
                3. 尽量不删减原文的内容，尽量克制对原文的增加
                4. 删除多余的换行符和空格 
                5. 保留所有引用和数据
                
                注意：
                - 仅对文本结构和连贯性进行优化
                - 不改变原文的表达风格和语气
                - 不添加新的观点或内容
                
                文本内容：
                {text}'''
            }
        ]
        
        try:
            response = Generation.call(
                model='qwen-max',
                messages=messages
            )
            return response.output.text.strip()
        except Exception as e:
            print(f"AI 增强处理失败: {e}")
            return text  # 如果处理失败，返回原文
    def format_directory(self, input_dir, output_dir=None):
        """批量处理目录下的所有 markdown 文件"""
        input_dir = Path(input_dir)
        if output_dir:
            output_dir = Path(output_dir)
        else:
            output_dir = input_dir / "ai_enhanced"
        
        # 创建输出目录
        output_dir.mkdir(exist_ok=True)
        
        # 获取所有 markdown 文件
        markdown_files = list(input_dir.glob("*.md"))
        total_files = len(markdown_files)
        
        print(f"\n=== 开始批量处理，共找到 {total_files} 个文件 ===")
        
        for i, file in enumerate(markdown_files, 1):
            try:
                print(f"\n处理文件 [{i}/{total_files}]: {file.name}")
                # 设置输出文件路径
                output_file = output_dir / f"{file.stem}_ai_enhanced.md"
                self._process_single_file(file, output_file)
            except Exception as e:
                print(f"处理文件 {file.name} 时出错: {e}")
        
        print(f"\n=== 批量处理完成 ===")
        print(f"输出目录: {output_dir}")
    
    def _process_single_file(self, input_file, output_file):
        """处理单个文件"""
        print(f"正在处理: {input_file.name}")
        
        # 1. 读取文件
        with open(input_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 2. 使用基础格式化
        base_formatted = self.paper_formatter.format_text(content)
        
        # 3. 分块进行AI增强处理
        chunks = self._split_into_chunks(base_formatted)
        enhanced_chunks = []
        
        for i, chunk in enumerate(chunks, 1):
            print(f"处理第 {i}/{len(chunks)} 块...")
            enhanced_chunk = self._enhance_with_ai(chunk)
            enhanced_chunks.append(enhanced_chunk)
        
        # 4. 合并处理后的内容
        enhanced_content = self._merge_chunks(enhanced_chunks)
        
        # 5. 保存结果
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(enhanced_content)
        
        print(f"完成: {output_file.name}")
    
    def _split_into_chunks(self, text, max_size=3000):
        """将文本按标题分割成块"""
        # 使用正则表达式匹配标题行
        title_pattern = r'^#+\s+.*$|^第[一二三四五六七八九十]+[章节].*$'
        
        # 按行分割文本
        lines = text.split('\n')
        chunks = []
        current_chunk = []
        current_size = 0
        
        for line in lines:
            # 检查是否是标题行
            if re.match(title_pattern, line.strip(), re.MULTILINE):
                # 如果当前块不为空，保存它
                if current_chunk:
                    chunks.append('\n'.join(current_chunk))
                    current_chunk = []
                    current_size = 0
                
            # 添加当前行到当前块
            current_chunk.append(line)
            current_size += len(line)
            
            # 如果当前块超过大小限制，且不是以标题开始，则分割
            if current_size > max_size and not re.match(title_pattern, current_chunk[0].strip()):
                chunks.append('\n'.join(current_chunk))
                current_chunk = []
                current_size = 0
        
        # 添加最后一个块
        if current_chunk:
            chunks.append('\n'.join(current_chunk))
        
        return chunks
    
    def _merge_chunks(self, chunks):
        """智能合并处理后的文本块"""
        merged_text = []
        
        for i, chunk in enumerate(chunks):
            if i > 0 and not chunk.startswith('#'):
                # 如果不是以标题开始，添加适当的分隔
                merged_text.append('\n\n')
            merged_text.append(chunk.strip())
        
        return '\n\n'.join(merged_text)
def main():
    formatter = AIEnhancedFormatter()
    
    # 设置输入和输出目录
    input_dir = r"E:\pdf文档转换\markdownoutput"
    output_dir = r"E:\pdf文档转换\markdownoutput\ai_enhanced"
    
    # 批量处理文件
    formatter.format_directory(input_dir, output_dir)

if __name__ == "__main__":
    main()