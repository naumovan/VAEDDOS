# VAEDDOS
В данном репозитории рассмотрено применение вариационного автоэнкодера (Variational AutoEncoder) в задачах детектирования аномалий в поведении пользователей сайта.
Доказана способность VAE детектировать такие аномалии поведения пользователей как DDOS, PortScan, чрезмерно длинные запросы.

Отличительными преимуществами VAE являются: 
1. Отсутствие необходимости обучения
2. Быстрота обработки
3. Возможность учитывать категориальные признаки

Подробнее почитать о VAE:
1. https://arxiv.org/pdf/1312.6114
2. https://arxiv.org/pdf/1906.02691
3. https://stepik.org/lesson/1305409/step/1?unit=1320365

##Структура репозитория
1. demonstarting - скрипт(ы) для демонстрации
2. data/train - данные используемые для обучения
3. train_script - ноутбук обучения модели (Warning: обучалось в Colab)
4. VAE - архитектура VAE для импортов и data_processing, в котором происходит работа обученной модели автоэнкодера
5. weights - обученная модель VAE

Данные взяты отсюда: https://www.kaggle.com/datasets/asfandyar250/network/data
