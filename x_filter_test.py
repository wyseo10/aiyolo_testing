#Filter will be choosen

from collections import deque

class MovingAverage:
    def __init__(self, window_size):
        self.window_size = window_size
        self.values = deque(maxlen=window_size)  # 큐의 최대 길이를 지정

    def update(self, new_value):
        self.values.append(new_value)  # 새로운 값을 추가
        return self.calculate_average()  # 평균을 반환

    def calculate_average(self):
        if len(self.values) == 0:
            return 0  # 값이 없으면 0 반환
        return sum(self.values) / len(self.values)  # 값들의 평균 계산

# Moving Average 객체 생성
window_size = 10  # 평균을 계산할 최근 값들의 개수 (10 fps 기준)
moving_average = MovingAverage(window_size)

# 시뮬레이션 예제: box_center_x 값이 업데이트되는 경우
import random
import time

for _ in range(20):  # 20번 값 업데이트 시뮬레이션
    box_center_x = random.uniform(0, 100)  # 0~100 사이의 랜덤 값 (임의의 box_center_x)
    stabilized_x = moving_average.update(box_center_x)  # 안정된 좌표 계산
    print(f"New Value: {box_center_x:.2f}, Stabilized Value: {stabilized_x:.2f}")
    time.sleep(0.1)  # 약 10fps로 업데이트되는 상황 가정