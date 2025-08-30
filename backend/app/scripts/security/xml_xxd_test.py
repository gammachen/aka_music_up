from asyncio import FastChildWatcher
from typing import no_type_check
from xml.dom.minidom import parseString
import xml.sax
from xml.sax.handler import ContentHandler
from xml.sax import make_parser
import os

os.environ['XML_CATALOG_FILES'] = 'file:///etc/passwd'
# 启用外部实体解析的函数(TODO 废弃，已废弃，使用xml.sar中的make_parser其实没有能够启用正确的外部实体，使用lxml来解析)
def parse_with_external_entities(xml_file_path):
    with open(xml_file_path, 'r') as file:
        xml_content = file.read()
    
    # 使用SAX解析器并启用外部实体
    parser = make_parser()
    # 启用外部实体解析
    parser.setFeature(xml.sax.handler.feature_external_ges, True)
    
    # 使用minidom解析
    dom = parseString(xml_content, parser)
    # dom = parser.parse(xml_file_path)
    return dom

# 假设这是一个接收用户上传XML文件的函数
def process_xml(xml_file_path):
    try:
        # 尝试使用启用外部实体的解析方式
        dom = parse_with_external_entities(xml_file_path)
        
        # 处理XML内容
        data = dom.getElementsByTagName('data')
        if data and data[0].firstChild:
            content = data[0].firstChild.nodeValue
            print(f"解析到的内容: {content}")
            
        cid_elements = dom.getElementsByTagName('CustomerID')
        if cid_elements and cid_elements[0].firstChild:
            cid_content = cid_elements[0].firstChild.nodeValue
            print(f"CustomerID: {cid_content}")
        
        # 解析Street元素内容
        street_elements = dom.getElementsByTagName('Street')
        if street_elements and street_elements[0].firstChild:
            street_content = street_elements[0].firstChild.nodeValue
            print(f"街道信息: {street_content}")
            return street_content
        else:
            print("未找到Street元素或元素为空")
            
    except Exception as e:
        print(f"解析XML时出错: {e}")
        import traceback
        traceback.print_exc()
        return None

# 调用解析函数
result = process_xml("/Users/shhaofu/Code/cursor-projects/aka_music/backend/app/scripts/security/user_input.xml")
print(f"返回结果: {result}")

# 另一种尝试方法 - 使用lxml库，它对外部实体有更好的支持
try:
    from lxml import etree
    print("\n尝试使用lxml解析:")
    
    # 创建一个启用DTD处理和外部实体解析的解析器
    parser = etree.XMLParser(resolve_entities=True)
    tree = etree.parse("/Users/shhaofu/Code/cursor-projects/aka_music/backend/app/scripts/security/user_input.xml", parser)
    
    root = tree.getroot()
    
    # 查找Street元素
    street = root.xpath('//Street')
    if street and len(street) > 0:
        print(f"lxml解析的街道信息: {street[0].text}")
        
    parser = etree.XMLParser(
        resolve_entities=True, 
        load_dtd=True, 
        no_network=False, # 允许网络访问
        # xml_catalog_files='file:///etc/passwd',
        dtd_validation=True)
    
    # 解析DTD文件
    '''
    with open(dtd_path, 'r') as dtd_file:
        dtd = etree.DTD(dtd_file)
        tree = etree.parse("/Users/shhaofu/Code/cursor-projects/aka_music/backend/app/scripts/security/user_input_dtd_send_message_sample.xml", parser)
        if dtd.validate(tree):
            print("XML文档符合DTD规范")    
    '''
    
    tree = etree.parse("/Users/shhaofu/Code/cursor-projects/aka_music/backend/app/scripts/security/user_input_dtd_send_message_sample.xml", parser)
    root = tree.getroot()
    ddata = root.xpath('//data')
    if ddata and len(ddata) > 0:
        print(f"lxml解析的data信息: {ddata[0].text}")
    
except ImportError:
    print("lxml库未安装，跳过此测试")
except Exception as e:
    print(f"使用lxml解析时出错: {e}")
