# Model
- 필요한 [data](https://drive.google.com/drive/folders/1Sky2lWnEuSsW4UizwLZkh6e__cY7UjSl)

## issue
1.**가로로 긴거는 상관없지만 세로로 이미지가 길면 인식이 잘 안된다. 프론트쪽이랑 상의가 필요할 듯 하다.**
   - 학습데이터의 높이(세로)가 28픽셀로 고정되어있고 28*28로 스케일을 맞춰서 숫자를 쪼개는거라서 높이(세로)가 높아지면 인식률이 떨어지는거같다?


2.**데이터를 행렬로 넘겨줄건지 이미지로 넘겨줄건지도 상의가 필요할 듯 하다.**

## calculation
Testing1
![test_result.PNG](./test/test_result.PNG)
<br></br>
Testing2
![test_result2.PNG](./test/test_result2.PNG)
<br></br>
Testing3
![test_result3.PNG](./test/test_result3.PNG)