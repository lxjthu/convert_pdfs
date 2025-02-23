import fitz
from pathlib import Path
from convert_pdfs import PDFToMarkdownConverter

def test_pdf_boundaries(pdf_files, pages=3):
    """测试多个PDF文件的页面边界和转换效果
    Args:
        pdf_files: PDF文件路径列表
        pages: 每个文件要测试的页数（默认3页）
    """
    # 创建转换器实例
    markdown_dir = Path(__file__).parent / "markdown_files"
    pdf_dir = Path(__file__).parent / "pdfs"
    converter = PDFToMarkdownConverter(pdf_dir, markdown_dir, debug=True)
    
    for pdf_path in pdf_files:
        print(f"\n开始处理文件：{pdf_path.name}")
        # 打开PDF并处理指定页数
        doc = fitz.open(pdf_path)
        try:
            markdown_content = []
            
            # 处理每个文件的前三页
            for page_num in range(min(pages, len(doc))):
                print(f"\n处理第 {page_num + 1} 页")
                page = doc[page_num]
                
                # 生成边界可视化图片
                converter.visualize_boundaries(page)
                
                # 获取页面内容
                page_height = page.rect.height
                blocks = page.get_text("dict")["blocks"]
                
                # 获取页面内容边界
                content_boundaries = converter.get_content_boundaries(blocks, page_height)
                
                # 提取图片和表格
                images = converter.extract_images(page, pdf_path.stem, content_boundaries)
                tables = converter.detect_tables(page)
                
                # 处理文本内容
                elements = []
                page_text = []
                
                for block in blocks:
                    if not converter.is_header_or_footer(block, page_height):
                        if "lines" in block:
                            y_pos = block["bbox"][1]
                            if converter.is_in_content_area(y_pos, content_boundaries):
                                text_content = []
                                for line in block["lines"]:
                                    for span in line["spans"]:
                                        text = span["text"].strip()
                                        if text:
                                            font_size = span["size"]
                                            processed_text = converter.detect_heading(text, font_size)
                                            text_content.append(processed_text)
                                if text_content:
                                    page_text.append({
                                        'content': '\n'.join(text_content),
                                        'y_pos': y_pos
                                    })
                
                if page_text:
                    # 按垂直位置排序文本块
                    page_text.sort(key=lambda x: x['y_pos'])
                    # 只进行基础合并，暂时跳过AI分析
                    merged_paragraphs = converter.merge_lines('\n'.join(item['content'] for item in page_text))
                    # 直接使用合并后的段落
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
                    table_content = converter.extract_table_content(page, table['bbox'])
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
                
                if page_num < min(pages, len(doc)) - 1:
                    markdown_content.append("\n---\n")
            
            # 保存前三页的转换结果
            final_content = converter.clean_text("\n".join(markdown_content))
            output_file = markdown_dir / f"{pdf_path.stem}_preview.md"
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(final_content)
            
            print(f"已生成预览文件：{output_file.name}")
            
        finally:
            doc.close()

if __name__ == "__main__":
    # 指定PDF文件目录
    pdf_dir = Path(__file__).parent / "pdfs"
    # 获取所有PDF文件
    pdf_files = list(pdf_dir.glob("*.pdf"))
    
    if not pdf_files:
        print("未找到PDF文件")
    else:
        print(f"找到 {len(pdf_files)} 个PDF文件")
        test_pdf_boundaries(pdf_files)