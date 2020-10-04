# Приспособление памяти агента к новым условиям

Ключевые слова: обучение с подкреплением, reinforcement learning, experience replay, оценка важности опыта, представление окружения.

## Аннотация
Алгоритмы experience replay позволяют агентам запоминать и переиспользовать опыт, полученный в предыдущих эпизодах процесса обучения. В рамках данной работы рассмотрены существующие алгоритмы experience replay, а именно, тривиальный experience replay, Prioritized Experience Replay,  Hindsight Experience Replay, Actor-Critic with experience replay, а также рассмотрен способ построения представления окружения с использованием вариационного автокодировщика и рекуррентной нейронной сети. Также проведен сравнительный анализ данных алгоритмов с точки зрения идей, лежащих в их основе, а также их применимости для решения задач обучения с подкреплением. В статье построен алгоритм experience replay, позволяющий агенту использовать информацию об окружении, с помощью которой он сможет корректировать свою политику для приспособления к новым условиям. В полученом алгритме используется оценка важности опыта, построенная на ошибке предсказания следующего состояния окружения с помощью рекуррентной сети, у которой на выходе сеть, моделирующая смесь гауссовых распределений.

## Введение
В задачах обучения с подкреплением (reinforcement learning) у агента нет никаких примеров правильного поведения для решения необходимой задачи, на которых он может обучаться, поэтому агент должен учиться основываясь на собственном опыте взаимодействия с окружением.

В данный момент в обучении с подкреплением остро стоит вопрос переиспользования в процессе обучения уже увиденного агентом опыта. Например, было бы полезно, чтобы беспилотные автомобили учились принимать решение, используя информацию о уже совершенных поездках. Однако, можно хранить лишь ограниченное количество информации, поэтому необходимо уметь хранить наиболее релевантную информацию.

Существующие алгоритмы experience replay решают описанные выше задачи, но их недостаток заключается в том, что они оценивают важность опыта, основываясь на внутренней политике агента, никак не учитывая информацию о новых условиях среды, в следствие чего агент не способен подстраиваться под изменения окружения.

Объектом исследования данной работы являются алгоритмы experience replay в задачах обучения с подкреплением.

В качестве предмета исследования выбрано применение алгоритмов experience replay для приспособления памяти агента к новым условиям среды.

Целью данной работы является построение алгоритма experience replay, использующего получаемый агентом опыт в соответствии с оценкой важности опыта, равной величине ошибки от предсказания будущего состояния окружения, и позволяющего агенту подстраиваться под изменения окружения.

Также выдвинуты следующие задачи исследования:

1. Провести обзор существующих алгоритмов experience replay в задачах обучения с подкреплением.
2. Построить алгоритм, предсказывающий дальнейшее состояние окружения.
3. Построить оценку важности получаемого агентом опыта в соответствии с ошибкой предсказыващего алгоритма.
4. Внедрить в существующий алгоритм experience replay построенную оценку важности опыта.

## Обзор предметной области
Обучение с подкреплением - область машинного обучения, занимающаяся задачами обучения агента посредством его взаимодействий с окружением.
В рамках данной статьи рассматриваются методы глубокого обучения с подкреплением для задач с одним агентом (single-agent deep reinforcement learning).


В момент времени ![equation](https://latex.codecogs.com/gif.latex?t) окружение находится в состоянии ![equation](https://latex.codecogs.com/gif.latex?s_t), агент может поменять состояние окружения, совершив действие ![equation](https://latex.codecogs.com/gif.latex?a_t). При этом в ответ на свое действие он получает награду ![equation](https://latex.codecogs.com/gif.latex?r_t), которая может быть как положительной, так и отрицательной. Цель агента - выбирать для взаимодействия с окружением действия так, чтобы они максимизировали его будущую дисконтированную суммарную награду. Для этого агент во время обучения выучиват собственную политику поведения. Политикой называется функция, отображающая можество состояний окружения в множество действий агента. В случае глубокого обучения с подкреплением в качестве политики используется нейронная сеть.


В ходе обучения агент исследует среду, затем оптимизирует свою политику, после чего использует полученный опыт. В ходе исследования окружения агент с вероятностью ![equation](https://latex.codecogs.com/gif.latex?%5Cvarepsilon) совершает случайное действие, а с вероятностью ![equation](https://latex.codecogs.com/gif.latex?1%20-%20%5Cvarepsilon) совершает действие, выбранное на основании текущей политики. Такая стратегия для изучения окружения называется ![equation](https://latex.codecogs.com/gif.latex?%5Cvarepsilon)-жадной. Во время этапа исследования окружения агент набирает опыт, который в дальнейшем используется для корректировки политики. А в ходе использования опыта агент использует только собственную политику для выбора действий, таким образом, на данном этапе проверяется, насколько хорошо агент научился принимать выбирать действия для достижения поставленной перед ним цели.

## Сравнение аналогов
Для выбора собственного метода решения проведем обзор уже существующих аналогов.

Был произведен поиск научных статей, в которых описаны алгоритмы experience replay, либо подходы к использованию информации от окружения. Поиск производился по ключевым словам "experience replay", "prioritized experience", "reinforcement learning", "обучение с подкреплением", "environment models" с использованием ресурсов Google Scholar и arXiv. Среди множества статей были выбраны опубликованные после 2013 года, представленные на международных конференциях NeurIPS или ICLR, а также с количеством цитирований не менее 100.

Ниже приведено описание аналогов, выбранных из результатов поиска.

### Тривиальный алгоритм experience replay (ER)
В тривиальном алгоритме experience replay [1] в буфер сохраняются переходы среды из состояния ![equation](https://latex.codecogs.com/gif.latex?s_t) в следующее состояние ![equation](https://latex.codecogs.com/gif.latex?s_%7Bt&plus;1%7D) с информацией о том, какое действие  ![equation](https://latex.codecogs.com/gif.latex?a_t) совершил агент и какую награду ![equation](https://latex.codecogs.com/gif.latex?r_t) при этом он получил в формате ![equation](https://latex.codecogs.com/gif.latex?%28s_t%2C%20a_t%2C%20r_t%2C%20s_%7Bt&plus;1%7D%29). При этом каждый элемент буфера может быть выбран для обучения с равной вероятностью.

### Prioritized Experience Replay (PER)
Prioritized Experience Replay [2] позволяет оценивать важность для обучения получаемого агеном опыта в задачах с дискретным пространством действий.
В данном случае используется буфер как в тривиальном алгоритме, но каждому переходу сопоставляется приоритет, вычисляемый в соответствии с temporal-difference error (TD-error) и обновляемый после каждого этапа обучения. При этом каждый элемент из буфера выбрается с вероятностью, пропорциональной его приоритету. Таким образом, для обучения будут использоваться переходы, наиболее полезные для агента на данном этапе. В данном случае переход считается тем более полезеным, чем больше он помогает минимизировать loss-функцию.

### Hindsight Experience Replay (HER)
Hindsight experience replay [3] позволяет обучать агента решать задачи в условиях редкой награды, без необходимости использования reward shaping (модификации награды с учетом предположений о том, как решать задачу, поставленную перед агентом).
Идея алгоритма заключается в использовании буфера как в тривиальном случае и в добавлении к каждому переходу цели, которой должен был достигнуть агент.  При этом каждый переход ![equation](https://latex.codecogs.com/gif.latex?s_t%20%5Crightarrow%20s_%7Bt&plus;1%7D) добавляется в буфер не только с изначальной целью, но и с подмножеством дополнительных целей, выбранных из множества всех состояний текущего эпизода. Такой подход позволяет получать дополнительные успешно завершившиеся эпизоды, которые обычно трудно самостоятельно получить агенту в условиях редкой награды.

### Actor-Critic with Experience Replay (ACER)
Алгоритм Actor-Critic with Experience Replay [4] использует идею сопоставления приоритета каждому переходу ![equation](https://latex.codecogs.com/gif.latex?s_t%20%5Crightarrow%20s_%7Bt&plus;1%7D). Но, в отличие от PER, позволяет решать задачи как с дискретным, так и с непрерывным пространством событий. Кроме того, в данной статье предприняты попытки решения дилеммы bias-variance для оценок приоритетов опыта агента.

### World models
В статье World Models [5] описывается подход, позволяющий строить представление окружения, которое дает возможность по текущему состоянию и действию агента предсказывать следующее состояние окружения.
В данном случае для предсказания следующего состояния окружения используется вариационный автокодировщик, который по исходному изображению состояния окружения получает его латентное представление [7], которое вместе с действием, совершенным агентом, передается рекуррентной нейронной сети с сетью смеси распределений [6] на выходе, предсказывающей следующее состояние.
Использование идей из данной статьи позволит агенту при принятии решений во время обучения использовать информацию об изменениях окружения. 

### Критерии сравнения аналогов

#### Использование оценки важности опыта
Использование оценки важности опыта позволяет выбирать наиболее релевантный опыт для каждого из этапов обучения, что позволяет агенту не тратить время на уже неактуальный, не помогающий обучению опыт. В будущем предполагается использование собственной построенной оценки важности опыта, поэтому предпочтительнее иметь готовый алгоритм experience replay, в который есть возможность внедрить собственную постренную оценку.

#### Тип пространства действий
В зависимости от постановки в задаче обучения с подкреплением может быть как дискретное, так и непрерывное пространство действий. Для разных типов пространств требуются разные алгоритмы обучения, поэтому предпочтительнее иметь универсальный алгоритм experience replay, не зависящий от конкретного типа используемого алгоритма обучения.

#### Использование информации об измненениях окружения
Использование информации об изменениях окружения позволяет агенту во время обучения приспосабливаться к новым условиям и в соответствии с этими корректировать свое поведение.

### Таблица сравнения по критериям

В таблице 1 приведены результаты сравнения аналогов по описанным выше критериям.

*Примечание:* В таблице для критерия "Тип пространства событий" значение "любой" означает, что алгоритм применим и для дискретного и для неперерывного пространств действий.

Таблица 1

|                                                  |   ER  |     PER    |  HER  |  ACER | World Models |
|--------------------------------------------------|:-----:|:----------:|:-----:|:-----:|:------------:|
| Использование оценки важности опыта              |   -   |      +     |   -   |   +   |       -      |
| Тип пространства событий                         | любой | дискретный | любой | любой |     любой    |
| Использование информации об изменениях окружения |   -   |      -     |   -   |   -   |       +      |

### Выводы по итогам сравнения
В результате сравнения аналогов можно сделать вывод о том, что наиболее подходящими для преследуемых целей являются алгоритмы PER и ACER, поскольку в данном случае есть возможность внедрения в них собственной оценки важности опыта. А непосредственно построение оценки важности опыта можно сделать на основе идей, изложенных в [5], поскольку они позволяют учитывать информацию об изменениях окружения, а также подходят для дискретных и непрерывных пространств событий.

## Выбор метода решения

В результате обзора аналогов было показано, что в существующих алгоритмах experience replay используется информация лишь об ошибках предсказания агента, основанных только на его собственной внутренней политике, и лишь в работе [5] представлен способ представления окружения, при котором возможно использование информации об изменениях в нем. Поэтому решение должно использовать идеи из [5] для реализации модели предсказания следующего состояния окружения, а также использовать ошибку предсказания модели в качестве оценки важности опыта.

Кроме того, в работах [2] и [4] представлены алгоритмы, основанные на оценках важности опыта, поэтому предпочтительнее, чтобы итоговое решение использовало один из данных алгоритмов в основе, поскольку имеется возможность внедрения в них своей оценки важности опыта.

При этом использование [4] вместе с идеями изложенными в [5] позволит реализовать алгоритм, подходящий для задач как с дискретным пространством действий, так и с непрепрерывным.

Разрабатываемый алгоритм experience replay должен удовлетворять следующим требованиям:

- В основе алгоритма лежит уже существующий алгоритм experience replay, использующий оценку важности опыта.
- В алгоритме в качестве состояния окружения используется не изображение, а его сжатое латентное представление
- Алгоритм реализует предсказание следующего состояния окружения
- Оценка важности опыта, используемая в алгоритме, построена на основании ошибки предсказания следующего состояния окружения. 

## Описание метода решения
В результате работы был построен алгоритм experience replay, использующий оценку важности опыта, построенную на основании информации об изменениях в окружении.

В основе построенного алгоритма лежат идеи для представления окружения, описанные в [5], а также алгоритм Prioritized Experience Replay, представленный в [2].

Изначально агент наблюдает состояние окружения в виде изображения, которое представляется многомерным числовым тензором. Однако возникает необходимость получать сжатое представление состояния окружения в виде одномерного числового тензора, который будет хранить только основную необходимую информацию. Эта обусловлено тем, что для обучения агента, как правило, не требуется полная и точная информация о состоянии, а достаточно лишь её обобщённой версии. 

Этого можно добиться, реализовав вариационный автокодировщик [7] - модель, которая состоит из двух нейронных сетей, называемыми кодировщиком и декодировщиком (схема представлена на рис. 1) Задача кодировщика получить для исходного изображения такое представление, называемое латентным, по которому затем декодировщик сможет восстановить исходное изображение с минимальными потерями. В таких условиях можно предполагать, что латентное представление содержит всю основную необходимую информацию об исходном изображении.

![vae](./pics/vae.png)
Рисунок 1 - Схема вариационного автокодировщика

Для предсказания следующего состояния окружения используется модель, состоящая из рекуррентной нейронной сети, совмещенной с сетью смеси распределений (далее - MDN-RNN). Таким образом, MDN-RNN принимает на вход латентное представление текущего состояния окружения, а также действие, совершенное агентом, а на выходе дает параметры смеси гауссовых распределений, из которой и берется предсказание следующего состояния среды.

Общая схема предсказания следующего состояния окружения представлена на рисунке 2.

![prediction](./pics/prediction.png)
Рисунок 2 - Схема предсказывания следующего состояния окружения


На рисунке 2 с помощью ![equation](https://latex.codecogs.com/gif.latex?Z_t) обозначено латентное представление состояния окружения в момент времени ![equation](https://latex.codecogs.com/gif.latex?t), с помощью ![equation](https://latex.codecogs.com/gif.latex?Z%27_%7Bt&plus;1%7D) - предсказание следующего состояния окружения, а ![equation](https://latex.codecogs.com/gif.latex?A_t) означает действие агента в момент времени ![equation](https://latex.codecogs.com/gif.latex?t).


В качестве оценки важности опыта, основанной на информации об изменениях окружения, используется среднеквадратичная ошибка предсказания следующего состояния окружения.
Данная оценка встроена в существующий алгоритм Prioritized Experience Replay путем использования её в качестве приоритета переходов состояний окружения.
Таким образом, предполагается, что когда в окружении происходит что-то новое и неожиданное для агента, ошибка предсказания следующего состояния будет большой, а соответственно и приоритет перехода будет выше. Это означает, что данный переход с большей вероятностью будет выбран из буфера для обучения, значит агент сможет быстрее использовать его и откорректировать свою политику.


## Заключение
В работе представлен обзор существующих алгоритмов experience replay в задачах обучения с подкреплением. Проведен сравнительный анализ данных подходов, в результате которого было выяснено, что лишь в [5] представлены идеи использования в ходе обучения информации об изменениях окружения. На основании сравнительного анализа были сформулированы требования к разрабатываемому алгоритму experience replay.

В результате выполнения работы был построен алгоритм experience replay для задач обучения с подкреплением, использующий получаемый агентом опыт в соответствии с величиной ошибки от предсказания будущего состояния окружения. В основе построенного алгоитма лежит алгоритм Prioritized Experience Replay, в которй внедрена собственная оценка приоритета опыта агента. 

Для построения оценки важности опыта используется рекуррентная нейронная сеть с сетью смеси распределений на выходе, использующая для предсказание действие агента, а также латентное представление состояния агента, полученное при помощи вариационного автокодировщика.

Недостатком полученного решения является то, что на данный момент алгоритм подходит только для задач, использующих дискретное пространство действий, поэтому в дальнейшем планируется в основе построенного алгоритма использовать идеи, изложенные в [4] для того, чтобы стало возможным и решение задач с непрерывным пространством действий.

В будущем планируется с помощью полученного алгоритма обучить агента на играх Atari и сравнить полученные результаты с state-of-the-art алгоритмами. Также планируется скомбинировать существующие методы experience replay и внедрить построенную оценку в полученные алгоритмы, после чего натренировать агента на играх Atari.