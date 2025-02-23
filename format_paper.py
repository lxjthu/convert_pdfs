import os
import dashscope
from dashscope import Generation
import re
import json

class PaperFormatter:
    def __init__(self):
        # 从环境变量获取 API key
        self.api_key = os.environ.get('DASHSCOPE_API_KEY')
        if not self.api_key:
            raise ValueError("请在环境变量中设置 DASHSCOPE_API_KEY")
        dashscope.api_key = self.api_key
        self.model = 'qwen-max'  # 可选: qwen-max, qwen-plus, qwen-turbo
    
    def set_model(self, model_name):
        """切换使用的模型"""
        available_models = ['qwen-max', 'qwen-plus', 'qwen-turbo']
        if model_name not in available_models:
            raise ValueError(f"不支持的模型。可用模型: {', '.join(available_models)}")
        self.model = model_name
        
    def _call_model(self, messages):
        """统一的模型调用接口"""
        response = Generation.call(
            model=self.model,
            messages=messages
        )
        return response.output.text
    def format_text(self, text):
        """处理文本格式"""
        print("\n=== 开始处理文本 ===")
        
        # 1. 分批处理长文本
        batches = self._split_into_batches(text)
        print(f"\n共分成 {len(batches)} 个批次")
        processed_batches = []
        
        for i, batch in enumerate(batches):
            print(f"\n处理第 {i+1} 个批次:")
            # 2. 合并错误换行
            merged_text = self._merge_lines(batch)
            
            # 3. 直接从文本中提取标题并立即格式化
            formatted_batch = self._process_batch_titles(merged_text)
            
            # 4. 调整图片位置
            formatted_batch = self._adjust_images(formatted_batch)
            processed_batches.append(formatted_batch)
        
        # 5. 合并所有处理后的批次
        result = self._merge_batches(processed_batches)
        print("\n=== 文本处理完成 ===")
        return result

    def _process_batch_titles(self, text):
        """处理单个批次中的标题"""
        print("\n正在提取并处理标题...")
        
        # 获取标题数据
        titles_data = self._extract_titles_from_text(text)
        if not titles_data:
            print("未获取到标题数据，返回原文")
            return text
        
        # 按标题长度排序，避免短标题替换长标题的一部分
        sorted_titles = sorted(titles_data, key=lambda x: len(x['text']), reverse=True)
        result = text
        
        for title in sorted_titles:
            original_text = title['text'].strip()
            level = title.get('level', 1)
            
            # 生成markdown格式的标题
            formatted = f"\n{'#' * level} {original_text}\n"
            
            # 在原文中定位并替换标题
            # 1. 先找到标题文本的位置
            title_pos = result.find(original_text)
            if title_pos == -1:
                print(f"警告: 未找到标题 '{original_text}'")
                continue
                
            # 2. 向前查找句子边界
            start = title_pos
            while start > 0 and result[start-1] not in '。！？\n':
                start -= 1
                
            # 3. 向后查找句子边界
            end = title_pos + len(original_text)
            while end < len(result) and result[end] not in '。，；：！？\n':
                end += 1
                
            # 4. 判断提取的文本是否确实包含标题
            context = result[start:end]
            if original_text in context:
                # 5. 只替换标题部分,保留前后文本
                before_text = result[start:title_pos]
                after_text = result[title_pos + len(original_text):end]
                new_text = before_text + formatted + after_text
                result = result[:start] + new_text + result[end:]
                print(f"成功处理标题: {original_text}")
        
        # 清理多余的换行（保留最多两个连续换行）
        result = re.sub(r'\n{3,}', '\n\n', result)
        return result
        


    # 可以删除以下不再需要的方法：
    # - _mark_titles_in_text
    # - _format_with_titles

    def _split_into_batches(self, text, batch_size=4000):
        """将文本分割成批次"""
        batches = []
        paragraphs = text.split('\n\n')
        current_batch = []
        current_size = 0
        
        for para in paragraphs:
            para_size = len(para)
            if current_size + para_size > batch_size and current_batch:
                # 当前批次达到大小限制，保存并开始新批次
                batches.append('\n\n'.join(current_batch))
                current_batch = [para]
                current_size = para_size
            else:
                # 添加段落到当前批次
                current_batch.append(para)
                current_size += para_size
        
        # 添加最后一个批次
        if current_batch:
            batches.append('\n\n'.join(current_batch))
        
        return batches
    
    def _merge_batches(self, batches):
        """智能合并处理后的批次"""
        # 检查标题的连续性
        merged_text = []
        for i, batch in enumerate(batches):
            if i > 0:
                # 检查是否需要添加分隔符
                if not batch.startswith('#') and not merged_text[-1].endswith('\n\n'):
                    merged_text.append('\n\n')
            merged_text.append(batch)
        
        return ''.join(merged_text)
    def _analyze_structure(self, text):
        """使用通义千问分析文章结构"""
        messages = [
            {
                'role': 'user',
                'content': f'''请分析这篇学术论文的结构，识别出:
                1. 主要章节和子章节
                2. 每个章节的层级关系
                3. 段落的逻辑分布
                请以JSON格式返回分析结果。
                
                论文内容:
                {text[:2000]}...'''
            }
        ]
        
        response = Generation.call(
            model=self.model,  # 使用设置的模型,
            messages=messages
        )
        
        return response.output.text
    
    def _extract_json_from_response(self, response):
        """从响应中提取 JSON 部分"""
        # 尝试查找 ```json 和 ``` 之间的内容
        json_match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
        if json_match:
            return json_match.group(1)
            
        # 如果没有找到代码块标记，尝试查找 [ 和 ] 之间的内容
        json_match = re.search(r'\[(.*?)\]', response, re.DOTALL)
        if json_match:
            return f'[{json_match.group(1)}]'
            
        return None
    def _extract_titles_from_text(self, text):
        """直接从文本中提取标题"""
        print("\n正在提取标题...")
        messages = [
            {
                'role': 'user',
                'content': f'''请直接从文本中提取所有可能的标题，标题通常具有以下特征：
                1. 标题识别特征：
                - 必须出现数字编号（如：1、1.1、一、（一）、（1）等）
                - 数字后面跟着文字内容
                - 文字可能被换行符或多个空格分隔
                - 与后续正文内容主题相关但语句不连贯
                - 可能出现在段落开头或单独成行
                
                2. 标题判断方法：
                - 检查数字编号后的文本是否概括了后续内容
                - 观察该行文本是否与上下文形成断裂
                - 判断是否存在层级关系（如：二级标题在一级标题下）
                - 数字编号之间存在顺承关系
                
                3. 返回格式要求：
                请返回JSON数组，每个标题对象格式如下：
                [
                    {{"text": "标题1", "level": 1}},
                    {{"text": "标题2", "level": 2}}
                ]
                
                以下是需要分析的文本：
                {text}'''
            }
        ]
        
        response = self._call_model(messages)
        try:
            # 先尝试从响应中提取 JSON 部分
            json_str = self._extract_json_from_response(response)
            if json_str:
                titles_data = json.loads(json_str)
                print(f"API返回标题数据: {json.dumps(titles_data, ensure_ascii=False, indent=2)}")
                # 直接返回解析后的标题数据
                return titles_data
            else:
                print("未找到有效的 JSON 数据")
                return text, {}
        except (json.JSONDecodeError, AttributeError) as e:
            print(f"解析标题数据失败: {e}")
            print(f"原始响应: {response}")
            return []  # 修改这里：返回空列表
    def _merge_lines(self, text):
        #合并错误的换行
        # 1. 使用正则表达式识别需要合并的行
        lines = text.split('\n')
        merged_lines = []
        
        for i in range(len(lines)):
            current_line = lines[i].strip()
            
            # 跳过空行
            if not current_line:
                merged_lines.append('')
                continue
                
            # 如果当前行以标点符号结尾，保留换行
            if re.search(r'[。！？]$', current_line):
                merged_lines.append(current_line)
                continue
                
            # 如果是标题行，保留换行
            if re.match(r'^#+\s+', current_line) or re.match(r'^第[一二三四五六七八九十]+[章节]', current_line):
                merged_lines.append(current_line)
                continue
                
            # 其他情况尝试合并行
            if i < len(lines) - 1:
                next_line = lines[i + 1].strip()
                if next_line and not re.match(r'^#+\s+', next_line):
                    merged_lines.append(current_line + next_line)
                    continue
                    
            merged_lines.append(current_line)
            
        return '\n'.join(merged_lines)
 

    def _replace_titles_in_content(self, content, formatted_titles):
        """在文档中替换标题"""
        result = content
        
        # 按长度排序标题，避免部分替换
        titles = sorted(formatted_titles.items(), key=lambda x: len(x[0]), reverse=True)
        
        for original, formatted in titles:
            result = result.replace(original.strip(), formatted.strip())
        
        return result
    def _adjust_images(self, text):
        """调整图片位置"""
        # 1. 提取所有图片标记
        images = re.findall(r'!\[.*?\]\(.*?\)', text)
        
        # 2. 将图片移动到适当位置
        formatted_text = text
        for img in images:
            # 找到图片相关的段落
            context = self._find_image_context(text, img)
            
            # 询问大模型合适的图片位置
            new_position = self._get_image_position(context, img)
            
            # 移动图片
            formatted_text = formatted_text.replace(img, '')
            formatted_text = self._insert_image(formatted_text, img, new_position)
            
        return formatted_text
    def _find_image_context(self, text, image):
        """找到图片上下文"""
        # 获取图片前后的文本内容
        img_pos = text.find(image)
        context_start = max(0, img_pos - 500)
        context_end = min(len(text), img_pos + 500)
        return text[context_start:context_end]
    def _get_image_position(self, context, image):
        """使用通义千问决定图片位置"""
        messages = [
            {
                'role': 'user',
                'content': f'''分析以下文本片段，为图片确定最合适的插入位置:
                1. 图片应该放在相关描述之后
                2. 避免打断段落的连贯性
                3. 只返回一个数字，表示应该插入的段落编号（从0开始计数）
                
                文本内容:
                {context}
                
                图片标记:
                {image}'''
            }
        ]
        
        response = Generation.call(
            model='qwen-max',
            messages=messages
        )
        
        # 尝试从响应中提取数字
        try:
            position = int(response.output.text.strip())
        except ValueError:
            # 如果无法获取有效的位置，默认放在文本末尾
            position = -1
            
        return position
    def _insert_image(self, text, image, position):
        """在指定位置插入图片"""
        # 根据大模型建议的位置插入图片
        lines = text.split('\n')
        lines.insert(int(position), f'\n{image}\n')
        return '\n'.join(lines)
    def process_large_text(self, text, batch_size=4000):
        """分批处理长文本"""
        # 将文本分成多个批次
        batches = []
        total_length = len(text)
        start = 0
        
        while start < total_length:
            end = start + batch_size
            # 找到合适的分割点（句号或段落结束）
            if end < total_length:
                # 向后查找最近的句号或换行符
                while end < total_length and text[end] not in ['。', '\n']:
                    end += 1
            batches.append(text[start:end])
            start = end
        
        # 处理每个批次
        processed_results = []
        for batch in batches:
            result = self._process_single_batch(batch)
            processed_results.append(result)
        
        # 合并结果
        return self._merge_results(processed_results)

    def _process_single_batch(self, batch_text):
        """处理单个批次的文本"""
        messages = [
            {
                'role': 'user',
                'content': f'''请处理以下文本片段，保持上下文连贯性：
                {batch_text}'''
            }
        ]
        
        response = Generation.call(
            model=self.model,  # 使用设置的模型
            messages=messages
        )
        
        return response.output.text

    def _merge_results(self, results):
        """合并处理结果"""
        # 简单的合并方式，直接拼接结果
        return ''.join(results)
def main():
    # 初始化时不再传入 api_key
    formatter = PaperFormatter()
    
    input_file = r"E:\pdf文档转换\markdown_files\“沉浸式动员”_乡村振兴中农村党组织动员农民的路径创新——基于两个村庄的案例分析.md"
    
    # 读取文件
    with open(input_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 格式化文本
    formatted_content = formatter.format_text(content)
    
    # 保存结果
    output_file = input_file.replace('.md', '_formatted.md')
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(formatted_content)

if __name__ == "__main__":
    main()


    """ 
    def _mark_titles_in_text(self, text, titles_info):
        #在原文中标记标题位置
        try:
            titles_data = json.loads(titles_info)
            result = text
            # 用于存储标记后的标题信息
            marked_titles = {}
            
            print("\n=== 开始标记标题 ===")
            print(f"原始标题数据: {json.dumps(titles_data, ensure_ascii=False, indent=2)}")
            
            # 为每个标题添加特殊标记
            for title in titles_data:
                original_text = title['text']
                context_before = title.get('context_before', '')
                context_after = title.get('context_after', '')
                level = title.get('level', 1)
                
                # 生成唯一标记
                marker = f"<<TITLE_{len(marked_titles)}_{level}>>"
                print(f"\n处理标题: {original_text}")
                print(f"生成标记: {marker}")
                
                # 在原文中定位并标记标题
                if context_before and context_after:
                    # 使用上下文精确定位
                    pattern = f"{re.escape(context_before)}({re.escape(original_text)}){re.escape(context_after)}"
                    print(f"使用上下文匹配模式: {pattern}")
                    result = re.sub(pattern, f"{context_before}{marker}\\1{marker}{context_after}", result)
                else:
                    # 使用标题文本定位
                    pattern = f"(?:^|\\n)(?:[0-9一二三四五六七八九十]+[.、]\\s*|（[^）]*）\\s*)?{re.escape(original_text)}(?=\\n|$)"
                    print(f"使用标题文本匹配模式: {pattern}")
                    result = re.sub(pattern, f"\n{marker}\\g<0>{marker}\n", result)
                
                # 保存标记信息
                marked_titles[marker] = {
                    'original': original_text,
                    'level': level
                }
                print(f"标记完成，标题: {original_text} -> 标记: {marker}")
            
            print("\n=== 标记过程完成 ===")
            print(f"共标记 {len(marked_titles)} 个标题")
            return result, marked_titles
        except json.JSONDecodeError as e:
            print(f"JSON 解析失败: {e}")
            return text, {}

    def _format_with_titles(self, content, titles):
        #使用格式化的标题重组文档
        print("\n=== 开始格式化标题 ===")
        print(f"原始内容前500字符: {content[:500]}")
        result = content
        
        # 按标题层级排序，确保正确的替换顺序
        sorted_titles = sorted(titles.items(), key=lambda x: (x[1]['level'], len(x[1]['original'])), reverse=True)
        print(f"待处理标题数量: {len(sorted_titles)}")
        
        for marker, title_info in sorted_titles:
            original = title_info['original']
            level = title_info['level']
            
            # 生成markdown格式的标题
            formatted = f"\n\n{'#' * level} {original.strip()}\n\n"
            print(f"\n处理标题: {original}")
            print(f"标记: {marker}")
            print(f"格式化后: {formatted}")
            
            # 检查标记是否存在于文本中
            if marker not in result:
                print(f"警告: 未找到标记 {marker}")
                continue
                
            # 替换标记的内容
            pattern = f"{marker}.*?{marker}"
            old_result = result
            result = re.sub(pattern, formatted, result, flags=re.DOTALL)
            
            # 检查是否成功替换
            if old_result == result:
                print(f"警告: 标题 '{original}' 替换失败")
    
        print(f"\n处理后内容前50字符: {result[:50]}")
        return result.strip()
"""