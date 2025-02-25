
import cv2
import numpy as np
from pathlib import Path
import fitz
from collections import deque
from sklearn.feature_extraction.text import TfidfVectorizer
import jieba
from openai import OpenAI
import os
from convert_pdfs import PDFToMarkdownConverter

class EnhancedPDFConverter:
    def __init__(self, pdf_dir, markdown_dir, debug=False):
        self.pdf_dir = Path(pdf_dir)
        self.markdown_dir = Path(markdown_dir)
        self.debug = debug
        self.context_stack = deque(maxlen=3)
        self.page_headers = {}
        self.page_footers = {}
        self.base_converter = PDFToMarkdownConverter(pdf_dir, markdown_dir, debug)
        self.page_width = 0  # 将在处理PDF时设置
        self.tfidf = TfidfVectorizer(
            tokenizer=lambda x: list(jieba.cut(x)),
            stop_words=['的', '了', '和', '与', '或', '在', '是']
        )
        # 更新为兼容模式的API地址
        self.client = OpenAI(
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            api_key=os.environ.get('DASHSCOPE_API_KEY'),
            timeout=30  # 添加超时设置
        )

    def debug_print(self, message):
        """调试信息打印"""
        if self.debug:
            print(message)

    def convert_all_pdfs(self):
        """转换目录下所有PDF文件"""
        self.markdown_dir.mkdir(exist_ok=True)
        for pdf_file in self.pdf_dir.glob('*.pdf'):
            self.convert_single_pdf(pdf_file)
    def analyze_layout(self, page):
        """使用OpenCV进行版面分析"""
        # 将PDF页面转换为图像
        pix = page.get_pixmap()
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
            pix.height, pix.width, 3)
        
        # 转换为灰度图
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        
        # 边缘检测
        edges = cv2.Canny(gray, 50, 150)
        
        # 查找轮廓
        contours, _ = cv2.findContours(
            edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 分析页眉页脚区域
        header_region = self._analyze_region(contours, pix.height, 'header')
        footer_region = self._analyze_region(contours, pix.height, 'footer')
        
        return header_region, footer_region
    
    def _analyze_region(self, contours, page_height, region_type):
        """分析特定区域（页眉或页脚）"""
        # 调整阈值：学术期刊的页眉页脚通常更窄
        threshold = page_height * 0.15  # 15%
        
        if region_type == 'header':
            # 筛选页眉区域的轮廓
            relevant_contours = [cnt for cnt in contours 
                            if cv2.boundingRect(cnt)[1] < threshold and
                            self._is_header_like(cnt)]
        else:
            # 筛选页脚区域的轮廓
            relevant_contours = [cnt for cnt in contours 
                            if cv2.boundingRect(cnt)[1] > page_height - threshold and
                            self._is_footer_like(cnt)]
                
        if relevant_contours:
            y_coords = [cv2.boundingRect(cnt)[1] for cnt in relevant_contours]
            return np.mean(y_coords)
        
        return threshold if region_type == 'header' else page_height - threshold

    def _is_header_like(self, contour):
        """判断轮廓是否符合页眉特征"""
        x, y, w, h = cv2.boundingRect(contour)
        # 1. 页眉通常较扁平
        if h > 30:  # 页眉高度通常不超过30像素
            return False
        # 2. 页眉通常横跨较大宽度
        if w < self.page_width * 0.2:  # 至少占页面宽度的20%
            return False
        return True

    def _is_footer_like(self, contour):
        """判断轮廓是否符合页脚特征"""
        x, y, w, h = cv2.boundingRect(contour)
        
        # 1. 页码区域特征
        if h <= 20 and w <= 100:  # 页码通常是小区域
            # 检查是否在页面的左、中、右位置
            left_aligned = x < self.page_width * 0.2
            center_aligned = abs(x + w/2 - self.page_width/2) < 100
            right_aligned = x + w > self.page_width * 0.8
            
            # 只要满足其中一个位置条件即可
            if left_aligned or center_aligned or right_aligned:
                return True
            
        # 2. 版权信息等其他页脚内容特征
        if h <= 30 and w >= self.page_width * 0.3:  # 横向较长的内容
            return True
            
        return False
    
    
    # def dynamic_context_management(self, page_text):
    #     """管理上下文信息"""
    #     # 提取关键词
    #     if not self.context_stack:
    #         self.tfidf.fit([page_text])
        
    #     # 获取TF-IDF特征
    #     tfidf_matrix = self.tfidf.transform([page_text])
    #     feature_names = self.tfidf.get_feature_names_out()
        
    #     # 获取最重要的关键词
    #     important_words = []
    #     for idx in tfidf_matrix.nonzero()[1]:
    #         if tfidf_matrix[0, idx] > 0.1:  # 设置阈值
    #             important_words.append(feature_names[idx])
        
    #     # 更新上下文堆栈
    #     context = {
    #         'text': page_text[:500],  # 保存前500个字符
    #         'keywords': important_words[:10]  # 保存前10个关键词
    #     }
    #     self.context_stack.append(context)
        
    #     return context

    def dynamic_context_management(self, page_text):
        """管理上下文信息"""
        # 更新上下文堆栈，直接使用全文
        context = {
            'text': page_text,  # 保存完整文本
            'preview': page_text[:200]  # 保存前200个字符用于调试显示
        }
        self.context_stack.append(context)
        return context

    def _process_chunk_with_context(self, chunk, context):
        """使用上下文处理文本块"""
        prompt = f"""请根据上下文分析并格式化以下文本段落：
        
        上下文全文：{context['text']}
        上下文预览：{context['preview']}
        
        当前文本：
        {chunk}
        
        请返回格式化后的Markdown文本。
        """
        
        try:
            response = self.client.chat.completions.create(
                model="qwen-turbo",
                messages=[
                    {'role': 'system', 'content': '你是一个专业的学术文献格式化助手'},
                    {'role': 'user', 'content': prompt}
                ]
            )
            return response.choices[0].message.content
        except Exception as e:
            self.debug_print(f"文本处理失败: {e}")
            return chunk  # 如果处理失败，返回原始文本
    
    def intelligent_chunking(self, page_text):
        """智能分块处理"""
        chunks = []
        max_chunk_size = 1000  # 根据模型token限制调整
        
        # 按段落分割
        paragraphs = page_text.split('\n\n')
        current_chunk = []
        current_size = 0
        
        for para in paragraphs:
            para_size = len(para)
            if current_size + para_size > max_chunk_size:
                # 当前块达到大小限制，保存并开始新块
                if current_chunk:
                    chunks.append('\n\n'.join(current_chunk))
                current_chunk = [para]
                current_size = para_size
            else:
                current_chunk.append(para)
                current_size += para_size
        
        # 添加最后一个块
        if current_chunk:
            chunks.append('\n\n'.join(current_chunk))
        
        return chunks
    
    def metadata_extractor(self, first_page):
        """提取文档元数据"""
        text = first_page.get_text()
        
        try:
            response = self.client.chat.completions.create(
                model="qwen-turbo",  # 使用通义千问模型
                messages=[
                    {'role': 'system', 'content': '你是一个专业的学术文献解析助手'},
                    {'role': 'user', 'content': f"""请从以下学术文本中提取并以JSON格式返回：
                    - 标题（title）
                    - 作者（authors，数组）
                    - 机构信息（institutions，数组）
                    - 期刊名称（journal）
                    - 摘要（abstract）
                    - 关键词（keywords，数组）
                    
                    文本内容：
                    {text[:2000]}
                    """}
                ]
            )
            return response.choices[0].message.content
        except Exception as e:
            self.debug_print(f"元数据提取失败: {e}")
            return None
    
    def convert_single_pdf(self, pdf_path):
        """增强版PDF转换处理"""
        try:
             # 1. 使用 PDFToMarkdownConverter 进行基础转换
            self.base_converter.convert_single_pdf(pdf_path)
            
            # 读取基础转换的结果
            base_output_file = self.markdown_dir / f"{pdf_path.stem}.md"
            with open(base_output_file, "r", encoding="utf-8") as f:
                base_content = f.read()
            
            # 2. 使用我们的增强功能处理
            doc = fitz.open(pdf_path)
            markdown_content = []
            
            # 1. 提取元数据
            metadata = self.metadata_extractor(doc[0])
            if metadata:
                markdown_content.append(metadata)
            
            # 2. 分析版面布局
            header_region, footer_region = self.analyze_layout(doc[0])
            self.header_footer_threshold = int(header_region)
            # 将基础转换的内容按段落分割
            base_paragraphs = base_content.split('\n\n')
            
            # 逐段处理
            for paragraph in base_paragraphs:
                if paragraph.strip():
                    # 更新上下文
                    context = self.dynamic_context_management(paragraph)  # 使用 paragraph 而不是 base_page_content
                    
                    # 智能分块（如果段落太长）
                    chunks = self.intelligent_chunking(paragraph)  # 使用 paragraph 而不是 base_page_content
                
                    # 处理每个文本块
                    for chunk in chunks:
                        processed_text = self._process_chunk_with_context(chunk, context)
                        markdown_content.append(processed_text)
                    
                    # 处理图片和表格
                    self._process_media_elements(page, pdf_path.stem, markdown_content)
                
                # 生成最终文档
                final_content = self.clean_text("\n".join(markdown_content))
            
            # 保存文件
            output_file = self.markdown_dir / f"{pdf_path.stem}.md"
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(final_content)
                
            print(f"Successfully converted {pdf_path.name} to {output_file.name}")
            
        except Exception as e:
            print(f"Error converting {pdf_path.name}: {str(e)}")
        finally:
            if 'doc' in locals():
                doc.close()
    
    def _process_chunk_with_context(self, chunk, context):
        """使用上下文处理文本块"""
        prompt = f"""请根据上下文分析并格式化以下文本段落：
        
        上下文全文：{context['text']}
        上下文预览：{context['preview']}
        
        当前文本：
        {chunk}
        
        请返回格式化后的Markdown文本。
        """
        
        try:
            response = self.client.chat.completions.create(
                model="qwen-turbo",
                messages=[
                    {'role': 'system', 'content': '你是一个专业的学术文献格式化助手'},
                    {'role': 'user', 'content': prompt}
                ]
            )
            return response.choices[0].message.content
        except Exception as e:
            self.debug_print(f"文本处理失败: {e}")
            return chunk
    
    def _process_media_elements(self, page, pdf_name, markdown_content):
        """处理页面中的媒体元素"""
        # 提取图片
        images = self.extract_images(page, pdf_name, 
                                  (self.header_footer_threshold, 
                                   page.rect.height - self.header_footer_threshold))
        
        # 提取表格
        tables = self.detect_tables(page)
        
        # 处理图片
        for img in images:
            markdown_content.append(f"\n![Figure {img['index']}]({img['path']})\n")
        
        # 处理表格
        for table in tables:
            table_content = self.extract_table_content(page, table['bbox'])
            markdown_content.append(f"\nTable {table['index']}:\n{table_content}\n")

def main():
    current_dir = Path(__file__).parent
    pdf_dir = current_dir / "pdfs"
    markdown_dir = current_dir / "markdownoutput"
    
    converter = EnhancedPDFConverter(pdf_dir, markdown_dir, debug=True)
    converter.convert_all_pdfs()

if __name__ == "__main__":
    main()