from pathlib import Path
from markitdown import MarkItDown

def test_pdf_conversion():
    # 设置路径
    current_dir = Path(__file__).parent
    pdf_dir = current_dir / "pdfs"
    output_dir = current_dir / "markdown_output"
    output_dir.mkdir(exist_ok=True)

    # 初始化 MarkItDown
    md = MarkItDown()

    # 遍历所有PDF文件
    for pdf_file in pdf_dir.glob('*.pdf'):
        print(f"\n正在处理: {pdf_file.name}")
        try:
            # 检查文件是否存在和可访问
            if not pdf_file.exists():
                print(f"文件不存在: {pdf_file}")
                continue
                
            print(f"文件大小: {pdf_file.stat().st_size / 1024:.2f} KB")
            print("开始转换...")
            
            # 转换PDF
            result = md.convert(str(pdf_file))
            
            # 保存转换结果
            output_file = output_dir / f"{pdf_file.stem}.md"
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(result.text_content)
            
            print(f"转换成功，已保存到: {output_file}")
            
        except Exception as e:
            print(f"转换失败: {str(e)}")
            print(f"错误类型: {type(e).__name__}")

if __name__ == "__main__":
    test_pdf_conversion()