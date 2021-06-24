from PIL import Image
from itertools import groupby
import tensorflow
import numpy as np

def pred(image_name):

    image = Image.open(image_name).convert("L")
    
    w = image.size[0] # 행길이
    h = image.size[1] # 열길이
    r = w / h # 행렬의 비율
    new_w = int(r * 28)
    new_h = 28
    new_image = image.resize((new_w, new_h))

    # 이미지 행렬화
    new_image_arr = np.array(new_image)
    new_inv_image_arr = 255 - new_image_arr
    final_image_arr = new_inv_image_arr / 255.0 

    m = final_image_arr.any(0)

    out = [final_image_arr[:,[*g]] for k, g in groupby(np.arange(len(m)), lambda x: m[x] != 0) if k]

    num_of_elements = len(out) # 몇개의 element가 있니?
    elements_list = []

    for x in range(0, num_of_elements):
        img = out[x]
    
        width = img.shape[1]
        filler = (final_image_arr.shape[0] - width) / 2
    
        if filler.is_integer() == False:    
            filler_l = int(filler)
            filler_r = int(filler) + 1
        else:                              
            filler_l = int(filler)
            filler_r = int(filler)
    
        arr_l = np.zeros((final_image_arr.shape[0], filler_l)) 
        arr_r = np.zeros((final_image_arr.shape[0], filler_r))
    
        help_ = np.concatenate((arr_l, img), axis= 1)
        element_arr = np.concatenate((help_, arr_r), axis= 1)
    
        element_arr.resize(28, 28, 1)

        elements_list.append(element_arr)
    
    
    elements_array = np.array(elements_list)
    elements_array = elements_array.reshape(-1, 28, 28, 1)

    # 모델 불러와서 예측
    model = tensorflow.keras.models.load_model('model.h5')
    elements_pred =  model.predict(elements_array)
    elements_pred = np.argmax(elements_pred, axis = 1)
    
    return elements_pred

# 계산기
def calculate(image_name):
    prediction = pred(image_name)
    
    elements = []
    
    for element in prediction:
        string_element = str(element)
        
        if string_element == '10':
            elements.append('/')
        elif string_element == '11':
            elements.append('+')
        elif string_element == '12':
            elements.append('-')
        elif string_element == '13':
            elements.append('*')
        else:
            elements.append(str(string_element))

    while True:
        try:
            calc = ''.join(map(str, elements))
            calc_answer = eval(calc)
            return (calc,'=', calc_answer)
            break
        
        except SyntaxError:
            print('계산할 수 없거나 인식이 안됩니다. 다시 시도해주세요.')
            print('인식 요소 :',elements)
            break
            