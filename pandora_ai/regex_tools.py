import os
_root_=os.path.dirname(os.path.abspath(__file__))
import sys
if not sys.path[0]==_root_:
    sys.path.insert(0,_root_)
def root_join(*args):
    return os.path.join(_root_,*args)

import re
import json

def init_document(string):
    return dict(context=(),tag='root',content=string,begin=0,end=len(string)-1)

def new_part(part, tag, start, end):
    return dict(
        context=(*part['context'], part['tag']),
        tag= tag,
        content= part['content'][start:end],
        begin= start + part['begin'],
        end= end + part['begin'] - 1
    )

def split_part_content(part, patterns):
    last_index = 0
    tagged_parts = []
    if patterns and not all([tag==part['tag'] for tag in patterns]):
        used_patterns={tag:pattern for tag,pattern in patterns.items() if tag!=part['tag']}
        combined_pattern = '|'.join(f"({pattern})" for pattern in used_patterns.values())
        matches = list(re.finditer(combined_pattern, part['content'], flags=re.DOTALL))

        for match in matches:
            match_start, match_end = match.span(0)

            # Add non-matched text as 'else' tag
            if match_start > last_index:
                tagged_parts.append(new_part(part, 'else', last_index, match_start))

            # Find the matched tag
            matched_tag = next(tag for i, tag in enumerate(used_patterns) if match.group(i + 1) is not None)

            # Add the matched part
            tagged_parts.append(new_part(part, matched_tag, match_start, match_end))
            last_index = match_end

    # Add any remaining non-matched text
    if last_index < len(part['content']):
        tagged_parts.append(new_part(part, 'else', last_index, len(part['content'])))

    return tagged_parts

def split_part(part, patterns):
    if part['tag'] != 'else':
        split_parts = split_part_content(part, patterns)
        
        content=[split_part(_part, patterns) for _part in split_parts]

        if len(content)>1 or content[0]['tag']!='else' or part['tag']=='root':
            part['content'] = content

    return part

def pack(part,processing_funcs=None):
    processing_funcs=processing_funcs or {}
    if isinstance(part['content'], str):
        content=part['content']
    elif isinstance(part['content'], list):
        content=''.join(pack(child,processing_funcs) for child in part['content'])
    tag=part['tag']
    context=part['context']
    if tag in processing_funcs:
        content=processing_funcs[tag](content,context)
    return content

def split(string,patterns):
    document=init_document(string)
    return split_part(document,patterns)
    
def process_regex(string,patterns,processing_funcs):
    document=split(string,patterns)
    return pack(document,processing_funcs)
        
if __name__=='__main__':

    # Example usage
    patterns = {
        'double_quote': r'"[^"]*"',
        'single_quote': r"'[^']*'"
    }

    processing_funcs = {
        'single_quote': lambda content,context: f"''{content}''" if not "double_quote" in context else content
    }

    text = r"""He said, "Hello, my name is 'John'!" and then whispered 'Goodbye'."""

    split_text = split(text,patterns)
    print(json.dumps(split_text))

    processed=process_regex(text,patterns,processing_funcs)
    print(processed)

