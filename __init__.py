import base64
note = base64.b64decode("44CQ6KeF6bG8QUnnu5jnlLvjgJHlpoLpnIDmm7TlpJrluK7liqnmiJbllYbliqHpnIDmsYIgK3Z4OiBtZWVleW8=").decode('utf-8')
class AnyType(str):
    def __ne__(self, __value: object) -> bool:
        return False
any_typ = AnyType("*")
from .meyo_node_Computational import  *
from .meyo_node_String import  *
from .meyo_node_File import *
from .meyo_node_Functional import *


NODE_CLASS_MAPPINGS = {

    #运算型节点：meyo_node_Computational
    "CompareInt": CompareInt,
    "FloatToInteger": FloatToInteger,
    "GenerateNumbers": GenerateNumbers,
    "GetRandomIntegerInRange": GetRandomIntegerInRange,
   

    #字符串处理：meyo_node_String
    "SingleTextInput": SingleTextInput,  
    "TextToList": TextToList,  
    "TextConcatenator": TextConcatenator,  
    "MultiParamInputNode": MultiParamInputNode,
    "NumberExtractor": NumberExtractor, 
    "AddPrefixSuffix": AddPrefixSuffix,
    "ExtractSubstring": ExtractSubstring,
    "ExtractSubstringByIndices": ExtractSubstringByIndices,
    "SplitStringByDelimiter": SplitStringByDelimiter,
    "ProcessString": ProcessString,
    "ExtractBeforeAfter": ExtractBeforeAfter,
    "SimpleTextReplacer": SimpleTextReplacer,  
    "ReplaceNthOccurrence": ReplaceNthOccurrence,
    "ReplaceMultiple": ReplaceMultiple,
    "BatchReplaceStrings": BatchReplaceStrings,
    "RandomLineFromText": RandomLineFromText,
    "CheckSubstringPresence": CheckSubstringPresence,
    "AddPrefixSuffixToLines": AddPrefixSuffixToLines,
    "ExtractAndCombineLines": ExtractAndCombineLines,
    "FilterLinesBySubstrings": FilterLinesBySubstrings,
    "FilterLinesByWordCount": FilterLinesByWordCount,
    "SplitAndExtractText": SplitAndExtractText,
    "CountOccurrences": CountOccurrences,
    "ExtractLinesByIndex": ExtractLinesByIndex,
    "ExtractSpecificLines": ExtractSpecificLines,
    "RemoveContentBetweenChars": RemoveContentBetweenChars,
    "ShuffleTextLines": ShuffleTextLines,
    "ConditionalTextOutput": ConditionalTextOutput,
    "TextConditionCheck": TextConditionCheck,
    "TextConcatenation": TextConcatenation,
    "ExtractSpecificData": ExtractSpecificData,
    "FindFirstLineContent": FindFirstLineContent,
    "GetIntParam": GetIntParam,
    "GetFloatParam": GetFloatParam,
    "GenerateVideoPrompt": GenerateVideoPrompt,

    #文件处理：meyo_node_File
    "LoadAndAdjustImage": LoadAndAdjustImage,
    "GenericImageLoader": GenericImageLoader,
    "ImageAdjuster": ImageAdjuster,
    "CustomCrop": CustomCrop,
    "SaveImagEX": SaveImagEX, 
    "FileCopyCutNode": FileCopyCutNode,   
    "FileNameReplacer": FileNameReplacer,    
    "WriteToTxtFile": WriteToTxtFile,   
    "FileDeleteNode": FileDeleteNode,   
    "FileListAndSuffix": FileListAndSuffix,
    "ImageOverlayAlignment": ImageOverlayAlignment,
    "TextToImage": TextToImage,

    "ReadExcelData": ReadExcelData,
    "WriteExcelData": WriteExcelData,
    "WriteExcelImage": WriteExcelImage,
    "FindExcelData": FindExcelData,
    "ReadExcelRowOrColumnDiff": ReadExcelRowOrColumnDiff,

    #功能型节点：meyo_node_Functional
    "GetCurrentTime": GetCurrentTime,
    "SimpleRandomSeed": SimpleRandomSeed,
    "SelectionParameter": SelectionParameter,
    "ReadWebNode": ReadWebNode,
    "DecodePreview": DecodePreview,     
}


NODE_DISPLAY_NAME_MAPPINGS = {

   #运算型节点：meyo_node_Computational
   "CompareInt": "Compare Numbers🐠meeeyo.com",
   "FloatToInteger": "Normalize Number🐠meeeyo.com",
   "GenerateNumbers": "Generate Number Range🐠meeeyo.com",
   "GetRandomIntegerInRange": "Random Integer In Range🐠meeeyo.com",

   #字符串处理：meyo_node_String
   "SingleTextInput": "Text Input🐠meeeyo.com",
   "TextToList": "Text To List🐠meeeyo.com",
   "TextConcatenator": "Text Concatenator🐠meeeyo.com",
   "MultiParamInputNode": "Multi-Param Input🐠meeeyo.com",
   "NumberExtractor": "Integer Parameters🐠meeeyo.com",
   "AddPrefixSuffix": "Add Prefix/Suffix🐠meeeyo.com",
   "ExtractSubstring": "Extract Between Tags🐠meeeyo.com",
   "ExtractSubstringByIndices": "Extract By Number Range🐠meeeyo.com",
   "SplitStringByDelimiter": "Split String By Delimiter🐠meeeyo.com",
   "ProcessString": "Process String🐠meeeyo.com",
   "ExtractBeforeAfter": "Extract Before/After🐠meeeyo.com",
   "SimpleTextReplacer": "Simple Text Replacer🐠meeeyo.com",
   "ReplaceNthOccurrence": "Replace Nth Occurrence🐠meeeyo.com",
   "ReplaceMultiple": "Replace Multiple Occurrences🐠meeeyo.com",
   "BatchReplaceStrings": "Batch Replace Strings🐠meeeyo.com",
   "RandomLineFromText": "Random Line From Text🐠meeeyo.com",
   "CheckSubstringPresence": "Check Substring Presence🐠meeeyo.com",
   "AddPrefixSuffixToLines": "Add Prefix/Suffix To Lines🐠meeeyo.com",
   "ExtractAndCombineLines": "Extract And Combine Lines🐠meeeyo.com",
   "FilterLinesBySubstrings": "Filter Lines By Substrings🐠meeeyo.com",
   "FilterLinesByWordCount": "Filter Lines By Word Count Range🐠meeeyo.com",
   "SplitAndExtractText": "Split And Extract Text🐠meeeyo.com",
   "CountOccurrences": "Count Occurrences🐠meeeyo.com",
   "ExtractLinesByIndex": "Extract Lines By Index🐠meeeyo.com",
   "ExtractSpecificLines": "Extract Specific Lines🐠meeeyo.com",
   "RemoveContentBetweenChars": "Remove Content Between Chars🐠meeeyo.com",
   "ShuffleTextLines": "Shuffle Text Lines🐠meeeyo.com",
   "ConditionalTextOutput": "Conditional Text Output🐠meeeyo.com",
   "TextConditionCheck": "Text Condition Check🐠meeeyo.com",
   "TextConcatenation": "Text Concatenation🐠meeeyo.com",
   "ExtractSpecificData": "Extract Specific Data🐠meeeyo.com",
   "FindFirstLineContent": "Find First Line Content🐠meeeyo.com",
   "GetIntParam": "Get Integer Param🐠meeeyo.com",
   "GetFloatParam": "Get Float Param🐠meeeyo.com",
   "GenerateVideoPrompt": "Generate Video Prompt Template🐠meeeyo.com",

   #文件处理：meyo_node_File
   "LoadAndAdjustImage": "Load & Adjust Image🐠meeeyo.com",
   "GenericImageLoader": "Generic Image Loader🐠meeeyo.com",
   "ImageAdjuster": "Image Adjuster🐠meeeyo.com",
   "CustomCrop": "Custom Crop🐠meeeyo.com",
   "SaveImagEX": "Save Image🐠meeeyo.com",
   "FileCopyCutNode": "File Operations🐠meeeyo.com",
   "FileNameReplacer": "Replace File Name🐠meeeyo.com",
   "WriteToTxtFile": "Write To TXT🐠meeeyo.com",
   "FileDeleteNode": "Delete/Clean Files🐠meeeyo.com",
   "FileListAndSuffix": "List Files And Suffix🐠meeeyo.com",
   "ImageOverlayAlignment": "Image Overlay Alignment🐠meeeyo.com",
   "TextToImage": "Text To Image🐠meeeyo.com",

   "ReadExcelData": "Read Excel Data🐠meeeyo.com",
   "WriteExcelData": "Write Excel Data🐠meeeyo.com",
   "WriteExcelImage": "Write Excel Image🐠meeeyo.com",
   "FindExcelData": "Find Excel Data🐠meeeyo.com",
   "ReadExcelRowOrColumnDiff": "Read Excel Row/Column Diff🐠meeeyo.com",
   
    #功能型节点：meyo_node_Functional
   "GetCurrentTime": "Current Time (timestamp)🐠meeeyo.com",
   "SimpleRandomSeed": "Random Integer🐠meeeyo.com", 
   "SelectionParameter": "Selection Parameter🐠meeeyo.com",
   "ReadWebNode": "Read Web Node🐠meeeyo.com",
   "DecodePreview": "Decode Preview🐠meeeyo.com",
}
