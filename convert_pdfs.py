import fitz  # PyMuPDF
import os
import re
from pathlib import Path
import hashlib
import numpy as np
import cv2
from PIL import Image
from PIL import ImageDraw
import io
from openai import OpenAI

class PDFToMarkdownConverter:
    def __init__(self, pdf_dir, markdown_dir, debug=False):
        self.pdf_dir = Path(pdf_dir)
        self.markdown_dir = Path(markdown_dir)
        self.markdown_dir.mkdir(parents=True, exist_ok=True)
        # 创建图片存储目录
        self.images_dir = self.markdown_dir / "images"
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.header_footer_threshold = 50
        # 用于追踪图片和表格编号
        self.image_counter = 1
        self.table_counter = 1
        # 调试开关
        self.debug = debug
        
    
    def debug_print(self, *args, **kwargs):
        """统一的调试信息打印函数"""
        if self.debug:
            print(*args, **kwargs)  

    def save_image(self, image_data, pdf_name):
        """保存图片并返回相对路径"""
        # 生成唯一的图片文件名
        image_hash = hashlib.md5(image_data).hexdigest()[:8]
        image_filename = f"{pdf_name}_img_{self.image_counter}_{image_hash}.png"
        image_path = self.images_dir / image_filename
        
        self.debug_print(f"保存图片：{image_filename}")
        
        # 保存图片
        with open(image_path, "wb") as f:
            f.write(image_data)
        
        # 返回相对路径（用于Markdown引用）
        return f"images/{image_filename}"

    def extract_images(self, page, pdf_name, content_boundaries):  # 添加 content_boundaries 参数
        """提取页面中的图片"""
        self.debug_print("\n=== 提取图片 ===")
        image_list = []
        img_info = page.get_images(full=True)
        self.debug_print(f"页面中发现 {len(img_info)} 个图片")
        
        for img_index, img in enumerate(img_info):
            try:
                xref = img[0]
                bbox = page.get_image_bbox(img)
                if bbox:
                    y_pos = (bbox[1] + bbox[3]) / 2
                    self.debug_print(f"图片 {img_index + 1} 位置: y={y_pos}")
                    if self.is_in_content_area(y_pos, content_boundaries):
                        base_image = page.parent.extract_image(xref)
                        if base_image:
                            image_data = base_image["image"]
                            image_path = self.save_image(image_data, pdf_name)
                            image_list.append({
                                'path': image_path,
                                'bbox': bbox,
                                'index': self.image_counter
                            })
                            self.image_counter += 1
            except Exception as e:
                self.debug_print(f"处理图片 {img_index + 1} 失败: {e}")
        
        return image_list

    def detect_tables(self, page):
        """检测页面中的表格"""
        self.debug_print("\n=== 检测表格 ===")
        tables = []
        rects = page.get_text("dict")["blocks"]
        
        for block in rects:
            if block.get("type") == "rect":
                bbox = block["bbox"]
                self.debug_print(f"检测到可能的表格边框：{bbox}")
                tables.append({
                    'bbox': bbox,
                    'index': self.table_counter
                })
                self.table_counter += 1
        
        self.debug_print(f"共检测到 {len(tables)} 个表格")
        return tables

    def extract_table_content(self, page, table_bbox):
        """提取表格内容并转换为Markdown格式"""
        self.debug_print("\n=== 提取表格内容 ===")
        x0, y0, x1, y1 = table_bbox
        self.debug_print(f"表格边界：({x0}, {y0}, {x1}, {y1})")
        
        table_text = page.get_text("text", clip=(x0, y0, x1, y1))
        rows = table_text.strip().split('\n')
        markdown_table = []
        
        if rows:
            headers = rows[0].split()
            self.debug_print(f"表头：{headers}")
            markdown_table.append('| ' + ' | '.join(headers) + ' |')
            markdown_table.append('| ' + ' | '.join(['---'] * len(headers)) + ' |')
            
            for row in rows[1:]:
                cells = row.split()
                markdown_table.append('| ' + ' | '.join(cells) + ' |')
        
        return '\n'.join(markdown_table)

    def detect_horizontal_line(self, block, page_height):
        """检测是否为水平分隔线，并判断是否在页眉页脚区域"""
        if block.get("type") == "rect":
            bbox = block["bbox"]
            x0, y0, x1, y1 = bbox
            width = x1 - x0
            height = y1 - y0
            
            if width > 100 and height < 2:
                if y0 < 200:  # header_threshold
                    self.debug_print(f"检测到页眉分隔线：y={y0}")
                    return True, 'header'
                elif y1 > (page_height - 100):  # footer_threshold
                    self.debug_print(f"检测到页脚分隔线：y={y1}")
                    return True, 'footer'
                else:
                    self.debug_print(f"检测到正文分隔线：y={y0}")
                    return True, None
        return False, None

    def get_content_boundaries(self, blocks, page_height):
        """通过分割线或距离来确定正文内容的边界"""
        self.debug_print("\n=== 确定页面边界 ===")
        header_line = None
        footer_line = None
        
        for block in blocks:
            is_line, position = self.detect_horizontal_line(block, page_height)
            if is_line:
                if position == 'header':
                    if header_line is None or block["bbox"][1] > header_line:
                        header_line = block["bbox"][1]
                elif position == 'footer':
                    if footer_line is None or block["bbox"][1] < footer_line:
                        footer_line = block["bbox"][1]
        
        if header_line is None:
            header_line = 90
            self.debug_print("使用默认页眉高度：90")
        
        if footer_line is None:
            footer_line = page_height - 100
            self.debug_print(f"使用默认页脚位置：{footer_line}")
        
        self.debug_print(f"最终边界：页眉={header_line}, 页脚={footer_line}")
        return header_line, footer_line

    def is_in_content_area(self, y_pos, content_boundaries):
        """判断某个位置是否在正文区域内"""
        start_y, end_y = content_boundaries
        buffer = 5
        result = (start_y + buffer) <= y_pos <= (end_y - buffer)
        self.debug_print(f"位置 y={y_pos} {'在' if result else '不在'}正文区域内")
        return result

    def merge_lines(self, text_blocks):
        """合并文本行，处理换行和分段"""
        self.debug_print("\n=== 合并文本行 ===")
        merged_text = []
        current_paragraph = []
        
        lines = text_blocks.split('\n')
        i = 0
        while i < len(lines):
            line = self.clean_text_line(lines[i])
            if not line:
                i += 1
                continue
                
            # 检查是否为数字编号行
            number_match = re.match(r'^__NUMBER_LINE__(\d+)__$', line)
            if number_match and i > 0 and i < len(lines) - 1:
                prev_line = merged_text[-1] if merged_text else ''
                next_line = self.clean_text_line(lines[i+1])
                
                # 如果上一行以标点结尾，且下一行存在
                if self.is_sentence_end(prev_line):
                    self.debug_print(f"合并数字编号 {number_match.group(1)} 与下一行")
                    # 将数字与下一行合并
                    if current_paragraph:
                        merged_text.append(''.join(current_paragraph))
                        current_paragraph = []
                    merged_text.append(prev_line)
                    current_paragraph.append(f"{number_match.group(1)}{next_line}")
                    i += 2  # 跳过下一行
                    continue
            

            # 如果是标题，直接添加并继续
            if self.is_title_or_list(line):
                if current_paragraph:
                    merged_text.append(''.join(current_paragraph))
                    current_paragraph = []
                merged_text.append(line)
                i += 1
                continue
            
            # 处理普通段落
            current_paragraph.append(line)
            
            # 判断是否需要结束当前段落
            if i + 1 < len(lines):
                next_line = self.clean_text_line(lines[i + 1])
                if next_line:
                    # 1. 下一行是标题
                    if self.is_title_or_list(next_line):
                        if current_paragraph:
                            merged_text.append(''.join(current_paragraph))
                            current_paragraph = []
                    # 2. 下一行有缩进（新段落开始）
                    elif next_line.startswith('  '):
                        if current_paragraph:
                            merged_text.append(''.join(current_paragraph))
                            current_paragraph = []
                    # 3. 当前行是句子结尾且下一行不是短句
                    elif self.is_sentence_end(line) and len(next_line) > 30:
                        if current_paragraph:
                            merged_text.append(''.join(current_paragraph))
                            current_paragraph = []
        
            i += 1
        
        # 处理最后一个段落
        if current_paragraph:
            merged_text.append(''.join(current_paragraph))
        
        return merged_text

    def clean_text_line(self, text):
        """清洗单行文本"""
        if not text:
            return ""
        
        # 1. 移除特殊格式标记
        text = re.sub(r'#[":]?#\s*', '', text)  # 移除 #"# 等标记
        text = re.sub(r'#{3,}', '', text)  # 移除多余的 ###
        # 1. 统一全角和半角字符
        text = text.replace('　', '  ')  # 全角空格转换为两个半角空格
        
        # 保存段落开始的缩进标记
        is_paragraph_start = text.startswith('  ')  # 现在只需要检查半角空格

        # 多个空格转换为1个空格
        if text.startswith('  ') or text.startswith('　'):
            text = '  ' + re.sub(r'\s+', ' ', text[2:])
        else:
            text = re.sub(r'\s+', ' ', text)
        
        # 2. 处理数字编号开头的标题格式
        title_match = re.match(r'^(\d+)\s*([^\d].*)$', text)
        if title_match:
            return f"{title_match.group(1)} {title_match.group(2).strip()}"
        
        
        # 3. 修复转义字符和保留必要的符号
        text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9[ \t\n\r\f\v],.:;?!()（）《》“”\"\'‘’「」『』、，。：；！？\x5b\x5d【】—.]+', '', text)
        
       # 4. 处理中文、数字和英文之间的空格
        text = re.sub(r'(?<=[\u4e00-\u9fa5])[ \t]+(?=[\u4e00-\u9fa5])', '', text)  # 移除中文之间的空格
        text = re.sub(r'(?<=[\u4e00-\u9fa5])[ \t]+(?=[a-zA-Z0-9])', '', text)  # 移除中文和英文/数字之间的空格
        text = re.sub(r'(?<=[a-zA-Z0-9])[ \t]+(?=[\u4e00-\u9fa5])', '', text)  # 移除英文/数字和中文之间的空格
        text = re.sub(r'(?<=\d)[ \t]+(?=\d)', '', text)  # 移除数字之间的空格
        
        # 5. 修复标点符号
        text = re.sub(r'([,.:;?!()（）《》\"\'“”‘’「」『』、，。：；！？\x5b\x5d【】—.])[ \t]+', r'\1', text)  # 移除标点后的空格
        text = re.sub(r'[ \t]+([,.:;?!()（）《》\"\'“”‘’「」『』、，。：；！？\x5b\x5d【】—.])', r'\1', text)  # 移除标点前的空格
        
        # 6. 统一空格
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()

        # 检查是否为引用数字行
        ref_number_match = re.match(r'^\s*\[(\d+)\]\s*$', text)
        if ref_number_match:
            self.debug_print(f"检测到引用数字：[{ref_number_match.group(1)}]")
            return f"__REF_NUMBER__{ref_number_match.group(1)}__"
        # 检查是否为数字编号行
        number_patterns = [
            r'^\s*(\d+)\s*$',           # 匹配单独的数字
            r'^\s*(\d+)\.\s*$',         # 匹配数字加点
            r'^\s*[（(](\d+)[)）]\s*$'   # 匹配带括号的数字
        ]
        
        for pattern in number_patterns:
            number_match = re.match(pattern, text)
            if number_match:
                self.debug_print(f"检测到数字编号：{text.strip()}")
                return f"__NUMBER_LINE__{number_match.group(1)}__"
        # 7. 恢复段落开始的缩进
        if is_paragraph_start and not title_match:  # 标题不添加缩进
            text = '  ' + text
        
        return text

    def clean_text(self, text):
        """最终清理文本"""
        # 1. 清理所有特殊标记
        text = re.sub(r'__NUMBER_LINE__(\d+)__', r'\1', text)  # 清理数字行标记
        text = re.sub(r'__REF_NUMBER__(\d+)__', r'[\1]', text)  # 清理引用标记
        
        # 2. 清理多余的换行和空格
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r' {2,}', ' ', text)
        return text.strip()
    
    def is_sentence_end(self, text):
        """判断是否是句子结尾"""
        end_punctuation = '。!！?？'
        return text and text[-1] in end_punctuation

    def is_title_or_list(self, text):
        """判断是否是标题或列表项"""
        if not text or len(text.strip()) == 0:
            return False
            
        # 清理文本，去除首尾空格
        text = text.strip()
        
        # 1. 检查数字编号格式（如：1.1、2.3.1）
        if re.match(r'^[\d\.]+\s+\S.*$', text):
            # 提取第一个数字，确保是合理的章节号（通常不会超过20章）
            first_num = int(re.match(r'^(\d+)', text).group(1))
            return first_num <= 20
        
        # 2. 检查中文数字编号（如：一、二、）
        if re.match(r'^[一二三四五六七八九十百千万]+[、．.]\s*\S.*$', text):
            return True
            
        # 3. 检查"第X章"、"第X节"等格式
        if re.match(r'^第[一二三四五六七八九十百千万\d]+[章节篇部]\s*\S.*$', text):
            return True
            
        # 4. 特定标题格式（如：摘要、引言、结论等）
        special_titles = {'摘要', '引言', '结论', '参考文献', '致谢', '附录'}
        if text in special_titles:
            return True
            
        # 5. 检查括号编号（如：（一）、(1)）
        if re.match(r'^[(（][一二三四五六七八九十\d][）)]\s*\S.*$', text):
            return True
            
        # 不再把所有短句都当作标题
        return False

    def visualize_boundaries(self, page):
        """可视化页面边界"""
        # 创建页面副本用于绘制
        pix = page.get_pixmap()
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        draw = ImageDraw.Draw(img)
        
        # 绘制水平参考线（每40像素一条）
        for y in range(0, 90, 10):  # 绘制前300像素的参考线
            draw.line([(0, y), (pix.width, y)], fill="red", width=2)
            # 添加标注
            draw.text((10, y+5), f"y={y}", fill="red")
        # 绘制页脚参考线（从底部向上每30像素一条）
        page_height = pix.height
        for y in range(page_height, page_height-100, -10):
            draw.line([(0, y), (pix.width, y)], fill="blue", width=2)
            draw.text((10, y-15), f"y={y}", fill="blue")
        # 保存调试图片
        debug_path = self.markdown_dir / "debug_images"
        debug_path.mkdir(exist_ok=True)
        img.save(debug_path / f"page_boundaries_{page.number}.png")
        self.debug_print(f"已保存边界可视化图片：page_boundaries_{page.number}.png")
    def convert_single_pdf(self, pdf_path):
        """转换单个PDF文件到Markdown"""
        try:
            doc = fitz.open(pdf_path)
            markdown_content = []
            
            for page_num in range(len(doc)):
                self.debug_print(f"\n=== 处理第 {page_num + 1} 页 ===")
                page = doc[page_num]
                # 添加可视化调试
                #self.visualize_boundaries(page)
                page_height = page.rect.height
                blocks = page.get_text("dict")["blocks"]
                
                # 获取页面内容边界
                content_boundaries = self.get_content_boundaries(blocks, page_height)
                
                # 提取图片和表格
                images = self.extract_images(page, pdf_path.stem, content_boundaries)
                tables = self.detect_tables(page)
                
                # 合并所有元素并按垂直位置排序
                elements = []
                page_text = []  # 收集当前页面的所有文本
                
                # 添加文本块
                for block in blocks:
                    if not self.is_header_or_footer(block, page_height):
                        if "lines" in block:
                            y_pos = block["bbox"][1]
                            if self.is_in_content_area(y_pos, content_boundaries):
                                text_content = []
                                for line in block["lines"]:
                                    for span in line["spans"]:
                                        text = span["text"].strip()
                                        if text:
                                            # 移除字体大小判断，直接添加文本
                                            text_content.append(text)
                                if text_content:
                                    page_text.append({
                                        'content': '\n'.join(text_content),
                                        'y_pos': y_pos
                                    })

                # 合并文本块
                if page_text:
                    # 按垂直位置排序文本块
                    page_text.sort(key=lambda x: x['y_pos'])
                    # 先进行基础合并
                    merged_paragraphs = self.merge_lines('\n'.join(item['content'] for item in page_text))
                    # 再进行AI分析合并
                
                # 处理收集到的文本
                if page_text:
                     # 按垂直位置排序文本块
                    page_text.sort(key=lambda x: x['y_pos'])
                    # 先进行基础合并
                    merged_paragraphs = self.merge_lines('\n'.join(item['content'] for item in page_text))
                    # 再进行AI分析合并
                    #final_paragraphs = self.merge_with_ai(merged_paragraphs)
                    # 添加到元素列表，直接使用列表中的段落
                    elements.extend([
                        {'type': 'text', 'content': paragraph, 'y_pos': page_text[0]['y_pos']}
                        for paragraph in merged_paragraphs
                    ])
                
                # 添加图片
                for img in images:
                    elements.append({
                        'type': 'image',
                        'content': f"\n![Figure {img['index']}]({img['path']})\n*Figure {img['index']}*\n",
                        'y_pos': img['bbox'][1]
                    })
                
                # 添加表格
                for table in tables:
                    table_content = self.extract_table_content(page, table['bbox'])
                    elements.append({
                        'type': 'table',
                        'content': f"\nTable {table['index']}:\n{table_content}\n",
                        'y_pos': table['bbox'][1]
                    })
                
                # 按垂直位置排序所有元素
                elements.sort(key=lambda x: x['y_pos'])
                
                # 将所有元素添加到markdown内容中
                for element in elements:
                    markdown_content.append(element['content'])
                
                #if page_num < len(doc) - 1:  # 最后一页不添加分隔符
                #    markdown_content.append("\n---\n")
            
            final_content = self.clean_text("\n".join(markdown_content))
            
            output_file = self.markdown_dir / f"{pdf_path.stem}.md"
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(final_content)
                
            print(f"Successfully converted {pdf_path.name} to {output_file.name}")
            
        except Exception as e:
            print(f"Error converting {pdf_path.name}: {str(e)}")
        finally:
            if 'doc' in locals():
                doc.close()

    # 其他方法保持不变...
    def is_header_or_footer(self, block, page_height):
        """判断文本块是否为页眉页脚"""
        y0 = block["bbox"][1]
        y1 = block["bbox"][3]
        
        if y0 < self.header_footer_threshold:
            return True
        if y1 > (page_height - self.header_footer_threshold):
            return True
        return False
    
    
    def detect_heading(self, text, font_size):
        return text

    def convert_all_pdfs(self):
        """转换目录中的所有PDF文件"""
        pdf_files = list(self.pdf_dir.glob("*.pdf"))
        
        if not pdf_files:
            print(f"No PDF files found in {self.pdf_dir}")
            return
        
        print(f"Found {len(pdf_files)} PDF files to convert")
        
        for pdf_file in pdf_files:
            self.convert_single_pdf(pdf_file)
        
        print("Conversion completed!")

def main():
    current_dir = Path(__file__).parent
    pdf_dir = current_dir / "pdfs"
    markdown_dir = current_dir / "markdownoutput"
    # 创建转换器实例时启用调试模式
    converter = PDFToMarkdownConverter(pdf_dir, markdown_dir, debug=True)
    converter.convert_all_pdfs()

if __name__ == "__main__":
    main()
            