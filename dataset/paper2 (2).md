# Объединение карт препятствий на базе теории Демпстера — Шафера.

Ключевые слова: TBM, карта препятствий(Occupancy grid map), слияние карт.

## Аннотация

Среди задач в робототехнике выделяют задачу одновременной локализации и построения карты. В данной статье приведено исследование различных подходов к объединению карт занятости. В ходе исследования были изучены пять различных способов слияния, базирующихся на различных подходах – адаптивного случайного блуждания, переносимой модели убеждений, генетического алгоритма, фильтра частиц, вероятностной обобщенной диаграмме Вороного. В качестве критериев были выбраны следующие характеристики: точность слияния, время обработки, поддержка параллелизма, тип ячейки(информация, хранимая внутри), использующийся в карте. В результате сравнения было выявлено, что у каждой из рассмотренных технологий есть слабые стороны в отношении рассмотренных критериев.  Однако использование генетического алгоритма позволяет выполнять задачу слияния быстрее других на картах одного размера. Но в этом алгоритме нет информации о качестве выполненного слияния. При этом, из указанных алгоритмов, наивысшую точность из известных имеет базирующийся на случайном блуждании.  Во втором аналоге нет информации, по которой можно было бы получить представление о качестве работы, однако ячейка карты в нем содержит максимальное количество информации. 

## Введение

Среди задач в робототехнике выделяют задачу одновременной локализации и построения карты. Одним из вариантов карт являются карты препятствий (Occupancy grid map). Использование таких карт целесообразно, когда с датчиков поступают зашумленные данные, либо робот нет априорной информации об окружающей среде. Такие карты используют автопилоты, роботы-спасатели и другие роботы, которые используют для перемещения информацию о местности вокруг себя. Построение карты при помощи данных только с одного робота происходит медленно и с ошибками, исправить которые затруднительно. Преодолеть эти недостатки можно, если использовать более одного робота, а карты, которые они строят, соединять в одну. Сложность заключается в том, чтобы исправлять конфликты, которые возникают при слиянии карт. 

Проблема заключается в том, что карты препятствий, полученные с помощью алгоритмов SLAM, в одном и том же окружении могут различаться. Появляется необходимость построения объединённой карты, полученной путем слияния нескольких исходных карт препятствий, обнаружение и устранение конфликтов на которой, является важной частью этой задачи.

Целью данной работы является разработка алгоритма решения конфликтов в картах препятствий на базе теории Демпстера—Шафера.  Объектом исследования являются двумерные карты препятствий,  а субъектом - конфликты при объединении двумерных карт препятствий. Для достижение поставленной цели необходимо было сформировать критерии и на их основе провести сравнительный анализ существующих решений, а так же создать алгоритм объединения.

## Обзор предметной области

В качестве аналогов отбирались статьи, описывающие объединение карт занятости, различающиеся по подходам слияния для возможности сравнения. Год выхода статей: 2005-2019.

Существуют следующие подходы к объединению карт:

##### 1. Merging occupancy grid maps from multiple robots[1]

Измерение сходства между картами используется для направления процесса поиска, который преобразует одну карту для достижения наибольшего перекрытия со второй. В алгоритме используется эвристика, основанная на специальной функции подобия изображения. Особенность функции в том, что она вычисляется за линейное время. Для процесса поиска используется адаптивное случайное блуждание (Adaptive random walk). Кроме того, вводиться специальная функция, которая указывает, было ли объединение успешным или нет. Качество объединения определяется по следующей формуле

<p align="center"><a href="https://www.codecogs.com/eqnedit.php?latex=ai(m_1,&space;m_2)=1-\frac{agr(m_1,m_2)}{agr(m_1,m_2)&plus;dis(m_1,m_2)}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?ai(m_1,&space;m_2)=1-\frac{agr(m_1,m_2)}{agr(m_1,m_2)&plus;dis(m_1,m_2)}" title="ai(m_1, m_2)=1-\frac{agr(m_1,m_2)}{agr(m_1,m_2)+dis(m_1,m_2)}" /></a></p>
где <a href="https://www.codecogs.com/eqnedit.php?latex=agr(m_1,m_2)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?agr(m_1,m_2)" title="agr(m_1,m_2)" /></a> - мера согласованности карт, а <a href="https://www.codecogs.com/eqnedit.php?latex=dis(m_1,m_2)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?dis(m_1,m_2)" title="dis(m_1,m_2)" /></a> - мера несогласованности. Если ячейка принадлежит только одной карте, то она не учитывается. Слияние ячеек происходит по следующему правилу:

* если информация с различных карт совпадают, то она записывается в ячейку результирующей карты,
* если информация есть только в одной карте, то берется значение из неё,
* если информация различается, то в результирующей карте данная ячейка помечается, как неизвестная.

##### 2. Credibilist Simultaneous Localization and Mapping with a LIDAR[2]

В работе алгоритм объединения базируется на теории переносимой модели убеждений (Transferable belief model[3]) для карт препятствий (Occupancy Grid Map). После нахождения относительного угла поворота и вектора трансляции слияние карт *A,B* в карту *D* происходит по конъюнктивному правилу:

<p align="center"><a href="https://www.codecogs.com/eqnedit.php?latex=\hat{m}^\Omega_{i,j}(D)=\sum_{A\cap&space;B=D}\hat{m}^\Omega_{i,j}(A)&space;\cdot&space;\tilde{m}^\Omega_{i,j}(B)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\hat{m}^\Omega_{i,j}(D)=\sum_{A\cap&space;B=D}\hat{m}^\Omega_{i,j}(A)&space;\cdot&space;\tilde{m}^\Omega_{i,j}(B)" title="\hat{m}^\Omega_{i,j}(D)=\sum_{A\cap B=D}\hat{m}^\Omega_{i,j}(A) \cdot \tilde{m}^\Omega_{i,j}(B)" /></a></p>
где <a href="https://www.codecogs.com/eqnedit.php?latex=m_{i,j}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?m_{i,j}" title="m_{i,j}"/></a> – ячейка карты препятствий с позицией *i,j*. А именно, данное правило применяется для каждой ячейки результирующей карты *D*

При определении согласованности карт используется *дизъюнктивный ортогональный оператор*(Disjunctive Orthogonal operator) с нормализацией конфликта, что, в конечном итоге, позволяет проигнорировать подвижный объект или артефакт.

<p align="center"><a href="https://www.codecogs.com/eqnedit.php?latex=Op_{OD}(\hat{m}^\Omega_{i,j},&space;\tilde{m}^\Omega_{i,j})&space;=&space;\frac{(\hat{m}^\Omega_{i,j}&space;\cup&space;\tilde{m}^\Omega_{i,j})(O)}{1-(\hat{m}^\Omega_{i,j}&space;\cap&space;\tilde{m}^\Omega_{i,j})(\emptyset)}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?Op_{OD}(\hat{m}^\Omega_{i,j},&space;\tilde{m}^\Omega_{i,j})&space;=&space;\frac{(\hat{m}^\Omega_{i,j}&space;\cup&space;\tilde{m}^\Omega_{i,j})(O)}{1-(\hat{m}^\Omega_{i,j}&space;\cap&space;\tilde{m}^\Omega_{i,j})(\emptyset)}" title="Op_{OD}(\hat{m}^\Omega_{i,j}, \tilde{m}^\Omega_{i,j}) = \frac{(\hat{m}^\Omega_{i,j} \cup \tilde{m}^\Omega_{i,j})(O)}{1-(\hat{m}^\Omega_{i,j} \cap \tilde{m}^\Omega_{i,j})(\emptyset)}" /></a></p>
##### 3. Multivehicle Cooperative Local Mapping[4]

В разделе слияния карт описывается целевая функция, основанная на вероятности занятости, и предоставлены некоторые конкретные процедуры, разработанные в духе генетического алгоритма для оптимизации целевой функции. Основываясь на этом методе, общим решением для ассоциации нескольких роботов служит дополнительно описанная стратегия непрямой оценки относительного положения автомобилей. Предложенный метод может выполнять задачу объединения, даже с большей начальной ошибкой выравнивания карты и высокой несогласованностью, присущей карте. Слияние происходит путем вычисления среднего значения исходных карт.

##### 4. Multi-robot Simultaneous Localization and Mapping using Particle Filter[5]

В данном подходе используется байесовская теория и фильтр частиц. Алгоритм работает в совокупности с задачей одновременной локализации и построения карты. Обновление карты в частице происходит по следующему правилу. 

<p align="center"><a href="https://www.codecogs.com/eqnedit.php?latex=m_t^{(i)}=M(z_t^1,x_t^{1(i)})&plus;M(z_t^2,x_t^{2(i)})&plus;M(\bar{z}_t^2,\bar{x}_t^{2(i)})&plus;m_{t-1}^{{(i)}}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?m_t^{(i)}=M(z_t^1,x_t^{1(i)})&plus;M(z_t^2,x_t^{2(i)})&plus;M(\bar{z}_t^2,\bar{x}_t^{2(i)})&plus;m_{t-1}^{{(i)}}" title="m_t^{(i)}=M(z_t^1,x_t^{1(i)})+M(z_t^2,x_t^{2(i)})+M(\bar{z}_t^2,\bar{x}_t^{2(i)})+m_{t-1}^{{(i)}}" /></a></p>
* <a href="https://www.codecogs.com/eqnedit.php?latex=M(z_t^1,&space;x_t^{1(i)})" target="_blank"><img src="https://latex.codecogs.com/gif.latex?M(z_t^1,&space;x_t^{1(i)})" title="M(z_t^1, x_t^{1(i)})"/></a>  – генератор карт, который возвращает частичную сетку занятости (выраженную в форме логарифмического правдоподобия, чтобы обеспечить линейную суперпозицию сеток).
* <a href="https://www.codecogs.com/eqnedit.php?latex=x_t^{1(i)}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?x_t^{1(i)}" title="x_t^{1(i)}"/></a> – последовательностью позиций робота 1.
* <a href="https://www.codecogs.com/eqnedit.php?latex=\bar{x}_t^{2(i)}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\bar{x}_t^{2(i)}" title="\bar{x}_t^{2(i)}" /></a>– последовательностью позиций робота 1, обращенная по времени.
* <a href="https://www.codecogs.com/eqnedit.php?latex=z_t^1" target="_blank"><img src="https://latex.codecogs.com/gif.latex?z_t^1" title="z_t^1" /></a> – последовательность наблюдений, измерений робота 1.
* <a href="https://www.codecogs.com/eqnedit.php?latex=\bar{z}_t^1" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\bar{z}_t^1" title="\bar{z}_t^1" /></a> – последовательность наблюдений, измерений робота 1, обращенная по времени.
* <a href="https://www.codecogs.com/eqnedit.php?latex=m_t^{(i)}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?m_t^{(i)}" title="m_t^{(i)}" /></a> – общая для частицы карта. 
* Оператор + – объединение двух карт. Поиск трансформации, и вычисление среднего.

Подобная обработка происходит в каждой частице. В последствии, по определенному правилу выбирается наилучшая частица и её карта считается конечной.

##### 5. A Topological Approach to Map Merging for Multiple Robots[6]

В этой статье обобщенная диаграмма Вороного (GVD) расширена для инкапсуляции вероятностной информации, закодированной в карте сетки занятости. Новая конструкция, называемая вероятностным GVD (PGVD), работает непосредственно с картами сетки занятости и используется для определения относительного преобразования между картами и их объединения. У этого подхода три основных преимущества: 

1. эффективен для нахождения относительных преобразований;
2. учитывается неопределенность, связанную с преобразованиями, используемыми для объединения карт
3. более определенные части карт предпочтительно используются в слиянии из-за вероятностного характера PGVD.

Процесс объединения представляет собой следующее. После нахождения относительного преобразования между двумя картами вероятности объединяются и фильтруются для получения окончательной карты. Данные, полученные в результате трансформации карты, включаются с использованием аддитивного свойства логарифмического представления занятости. Энтропийный фильтр применяется к объединенной карте. Фильтр энтропии сравнивает исходную и объединенную карты и отклоняет обновления, которые приводят к более высокой энтропии. Взаимная информация, определяется как уменьшение энтропии в ячейке *(i,j)* между исходной картой и объединенной картой. Конечная карта, определяется следующим образом: если значение взаимной информации для ячейки неотрицательно, то берутся данные из объединенной карты, иначе - из исходной.

#### Критерии

* **Точность** - качество наложения пересекающихся частей карт, измеряется в **%**. Критерий необходим, так как точность объединения карт является основным показателем качества работы алгоритма. Точность может быть измерена путем анализа результирующей карты и сравнения с исходными. В части рассмотренных алгоритмах точность либо не измеряется, либо не указывается. В статьях, в которых точность указана, методы измерения различаются между собой.
* **Время обработки**. Время обработки карта, площадью около 1600 м^2. Скорость важна, но только в с случаях объединения карт в задачах реального времени.
* **Поддержка параллелизма** - позволяет ускорить или улучшить качество объединения.
* **Тип ячейки** - информация, хранимая в ячейке. Тип ячейки важен, поскольку на базе ячейки строится оценка качества. Возможными типами могут являться:
  * {-1,0,1} - одно из трех целых чисел. "-1" обозначает, что ячейка свободна, "0" - нет информации о данной ячейке, "1" - ячейка занята.
  * TBM - ячейка содержит информацию в формате переносимой модели убеждений.
  * [0,1] - вещественное число, лежащее в промежутке от 0 до 1, обозначающее вероятность занятости данной ячейки.  "0" обозначает, что ячейка свободна, "0.5" - нет информации о данной ячейке, "1" - ячейка занята. 

Результаты сравнения рассмотренных аналогов по выбранным критериям представлены в таблице 1.

Таблица 1 - сравнение аналогов.

|                        | 1        | 2    | 3     | 4     | 5     |
| ---------------------- | -------- | ---- | ----- | ----- | ----- |
| Точность, %            | 97       | -    | -     | -     | 92    |
| Время обработки, с     | 170      | -    | 15    | 173   | 34    |
| Поддержка параллелизма | -        | -    | +     | +     | -     |
| Тип ячейки             | {-1,0,1} | TBM  | [0,1]​ | [0,1]​ | [0,1]​ |

В результате сравнения было выявлено, что у каждой из рассмотренных технологий есть слабые стороны в отношении рассмотренных критериев.  Однако использование генетического алгоритма(3) позволяет выполнять задачу слияния быстрее других на картах одного размера. Но в этом алгоритме нет информации о качестве выполненного слияния. При этом, из указанных алгоритмов, наивысшую точность из известных имеет базирующийся на случайном блуждании(1).  Во втором аналоге нет информации, по которой можно было бы получить представление о качестве работы, однако ячейка карты в нем содержит максимальное количество информации. Такой подход является самым информативным из аналогов.

## Выбор метода решения

Решение должно представлять собой алгоритм, выполняющий объединение карт препятствий и обладать следующими качествами:

* Ячейка типа TBM(transferable belief model, переносимая модель убеждений). Данным свойством обладает аналог *Credibilist Simultaneous Localization and Mapping with a LIDAR*. Так как количество информации, которое содержится в ячейке, важно для определения качества слияния карт, ячейка должна представлять собой элемент теории TBM. 
* Приемлемое время работы. Наиболее быстрая из предложенных обработка выполняется алгоритмом, предложенным в *Multivehicle Cooperative Local Mapping*, базирующимся на генетическом подходе. Однако скорость обработки информации не так важна, поскольку объединение карт не предполагается выполнять в задачах реального времени. Под приемлемым временем работы подразумевается обработка карт площадью 1600 м^2 не дольше 10 секунд.
* Точность слияния. В каждом из рассмотренных аналогов точность считается индивидуальным способом, отличающимся от других, или не считается вообще. В данном алгоритме точность слияния должна быть не менее 90%.
* Отсутствие поддержки параллелизма. Причиной отказа является увеличение вычислительной сложности.
* Другим преимуществом будет являться отсутствие вычислительно сложно математики, такой как вычисление обратных матриц.

## Общие положения TMB

Дано множество <a href="https://www.codecogs.com/eqnedit.php?latex=\Omega" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\Omega" title="\Omega" /></a> из *N* гипотез, так же называемых структурными различиями.

Булеан:
<p align="center"><a href="https://www.codecogs.com/eqnedit.php?latex=\Omega&space;=&space;\{H_n\}\space&space;\forall&space;n&space;\in[1,N]$\\&space;2^\Omega&space;=&space;\mathcal{P}(\Omega)&space;=&space;{A\mid&space;A\subseteq\Omega}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\Omega&space;=&space;\{H_n\}\space&space;\forall&space;n&space;\in[1,N]$\\&space;2^\Omega&space;=&space;\mathcal{P}(\Omega)&space;=&space;{A\mid&space;A\subseteq\Omega}" title="\Omega = \{H_n\}\space \forall n \in[1,N]$\\ 2^\Omega = \mathcal{P}(\Omega) = {A\mid A\subseteq\Omega}" /></a></p>
В TBM объединение гипотез <a href="https://www.codecogs.com/eqnedit.php?latex=H_n&space;\cup&space;H_m&space;(\forall&space;n,m&space;\in&space;[1,N],&space;n&space;\neq&space;m)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?H_n&space;\cup&space;H_m&space;(\forall&space;n,m&space;\in&space;[1,N],&space;n&space;\neq&space;m)" title="H_n \cup H_m (\forall n,m \in [1,N], n \neq m)" /></a> описывает отсутствие знаний между этими двумя гипотезами и элементом <a href="https://www.codecogs.com/eqnedit.php?latex=\emptyset" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\emptyset" title="\emptyset" /></a> , называемое *конфликтом* - представляет собой часть противоречивой информации между источниками.

По сравнению с базовой вероятностной теорией, в которой <a href="https://www.codecogs.com/eqnedit.php?latex=p(h)&space;&plus;&space;p(\overline{h})&space;=&space;1" target="_blank"><img src="https://latex.codecogs.com/gif.latex?p(h)&space;&plus;&space;p(\overline{h})&space;=&space;1" title="p(h) + p(\overline{h}) = 1" /></a>, в TBM сводится к булеану <a href="https://www.codecogs.com/eqnedit.php?latex=2^\Omega&space;=&space;\{h,\overline{h},&space;\Omega_{h&space;\overline{h}},&space;\O_{h&space;\overline{h}}\}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?2^\Omega&space;=&space;\{h,\overline{h},&space;\Omega_{h&space;\overline{h}},&space;\O_{h&space;\overline{h}}\}" title="2^\Omega = \{h,\overline{h}, \Omega_{h \overline{h}}, \O_{h \overline{h}}\}" /></a>.  Соответственно состояние описывается 4мя массами: <a href="https://www.codecogs.com/eqnedit.php?latex=m^\Omega(h),&space;m^\Omega(\overline{h}),&space;m^\Omega(\Omega),&space;m^\Omega(\O)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?m^\Omega(h),&space;m^\Omega(\overline{h}),&space;m^\Omega(\Omega),&space;m^\Omega(\O)" title="m^\Omega(h), m^\Omega(\overline{h}), m^\Omega(\Omega), m^\Omega(\O)" /></a>. Эти четыре массы составляют Basic Belief Assignment, обозначаемое <a href="https://www.codecogs.com/eqnedit.php?latex=m^\Omega" target="_blank"><img src="https://latex.codecogs.com/gif.latex?m^\Omega" title="m^\Omega" /></a>: 

<p align="center"><a href="https://www.codecogs.com/eqnedit.php?latex=\sum_{A\in&space;2^\Omega}m^\Omega(A)&space;=&space;1" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\sum_{A\in&space;2^\Omega}m^\Omega(A)&space;=&space;1" title="\sum_{A\in 2^\Omega}m^\Omega(A) = 1" /></a></p>
Недостаток знаний или неизвестность между <a href="https://www.codecogs.com/eqnedit.php?latex=h" target="_blank"><img src="https://latex.codecogs.com/gif.latex?h" title="h" /></a> и <a href="https://www.codecogs.com/eqnedit.php?latex=\overline{h}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\overline{h}" title="\overline{h}" /></a> явно описываются массой <a href="https://www.codecogs.com/eqnedit.php?latex=m^\Omega(\Omega)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?m^\Omega(\Omega)" title="m^\Omega(\Omega)" /></a>, и если два источника дают противоречивую информацию, то масса конфликта <a href="https://www.codecogs.com/eqnedit.php?latex=m^\Omega(\emptyset)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?m^\Omega(\emptyset)" title="m^\Omega(\emptyset)" /></a> увеличивается.

## Описание метода решения

Для решения задачи слияния двух карт нам необходимо иметь информация из ячейки карты, соответствующую переносимой модели убеждений. Карта представляет собой двумерный массив таких ячеек. Каждая ячейка является некоторым вектором действительных чисел, размерности 4. Значения этого вектора являются вероятностями того, что данная ячейка свободна, занята, неопределенна, в конфликте.

Разрабатываемый алгоритм предполагает, что информация об относительной трансформации карт (поворот и смещение) известна. По этой информации выполняется относительное выравнивание карт. После чего для каждой пары ячеек из первой и второй карты с соответствующими координатами применяется *disjunctive rule*:

<p align="center"><a href="https://www.codecogs.com/eqnedit.php?latex=m^\Omega_{1\cup&space;2}(C)&space;=&space;\sum_{A\cup&space;B=C}&space;m_1^\Omega(A)m_2^\Omega(B),&space;\forall&space;A,B,C&space;\subset&space;2^\Omega" target="_blank"><img src="https://latex.codecogs.com/gif.latex?m^\Omega_{1\cup&space;2}(C)&space;=&space;\sum_{A\cup&space;B=C}&space;m_1^\Omega(A)m_2^\Omega(B),&space;\forall&space;A,B,C&space;\subset&space;2^\Omega" title="m^\Omega_{1\cup 2}(C) = \sum_{A\cup B=C} m_1^\Omega(A)m_2^\Omega(B), \forall A,B,C \subset 2^\Omega" /></a></p>
Особенностью данного правила является то, что в результирующей карте сохраняются только согласованные между первой и второй картой значения. Т. е. если в одной из карт нет информации о данной ячейке, или значения сильно различаются (в одной карте ячейка занята, а во второй - свободна), то в результирующей карте большая часть массы будет сосредоточена у состояния $\Omega$. Полученная информация будет являться каркасом результирующей карты.

## Возможное дальнейшее развитие алгоритма

Одним из планов является преобразование объединяемых карт в изображение, на которых выделяются ключевые точки и вычисляются дескрипторы. С их помощью будут находиться области, которые присутствуют на одной карте и отсутствуют на другой. Данные области будут переноситься в результирующую карту с поправкой масс. Другим направлением исследования является попытка расширения дескрипторов с помощью TBM. Еще одно направление - поиск кластеров особых точек с с последующей обработкой.

## Заключение

В данной работе представлен сравнительный анализ подходов для решения задачи объединения карт занятости. Были сформированы критерии и описаны существующие решения. Полученный обзор позволяет оценить существующие методы в рамках выбранных критериев. Так же было предложена некоторая база алгоритма объединения, которую улучшать одним из предложенных или другим способом.

## Список литературы

1. Birk A., Carpin S. Merging occupancy grid maps from multiple robots //Proceedings of the IEEE. – 2006. – Т. 94. – №. 7. – С. 1384-1397.
2. Trehard G. et al. Credibilist simultaneous localization and mapping with a lidar //2014 IEEE/RSJ International Conference on Intelligent Robots and Systems. – IEEE, 2014. – С. 2699-2706.
3. Smets P. Data fusion in the transferable belief model //Proceedings of the third international conference on information fusion. – IEEE, 2000. – Т. 1. – С. PS21-PS33 vol. 1.
4. Li H. et al. Multivehicle cooperative local mapping: A methodology based on occupancy grid map merging //IEEE Transactions on Intelligent Transportation Systems. – 2014. – Т. 15. – №. 5. – С. 2089-2100.
5. Howard A. Multi-robot simultaneous localization and mapping using particle filters //The International Journal of Robotics Research. – 2006. – Т. 25. – №. 12. – С. 1243-1256.
6. Saeedi S. et al. Group mapping: A topological approach to map merging for multiple robots //IEEE Robotics & Automation Magazine. – 2014. – Т. 21. – №. 2. – С. 60-72.