#qwen_vl_utils.py: this file contains a utility function thats specific to the Qwen Vision model
# this fucntion serves as a brige between the message format and what the model expects as input
# it extract images from structured messages providing a clean abstraction for the model processing 
from typing import List,Dict, Tuple
from PIL import Image 
import base64
import io 

def_process_vision_info(messages:List[Dict])-> Tuple[List[Image.Image], List]:
    ""Proecess vision information from messages""

    images = []
    videos = []

    for message in messages: 
        if "content" in message: 
            for  content_item in message ["content"]:
                 if content_item.get("type") == "image":
                     if isinstance(image,Image.Image):
                         images.append(image)
    return images, videos # we only support images but since its a multimodal model we can just keep this (incase we wanted to support video @ sm pnt)