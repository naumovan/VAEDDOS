import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('../data/traffic_test.csv')

plt.figure(figsize=(12, 6))
# График битрейта от времени
plt.plot(data['No.'], data['Length'])
plt.xlabel('No.')
plt.ylabel('Length (КБ/сек)')
plt.title('Изменение Length от запроса к запросу')

# Отображение графиков
plt.tight_layout()
plt.show()
