# Исследование и поиск оптимального сочетания параметров алгоритмов вытеснения для применения в параллельном логструктурированном SSD кэше
Ключевые слова: SSD кэш, срок службы, алгоритм вытеснения, логструктурированный кэш, параллельный кэш.
## Аннотация
Статья посвящена актуальной сегодня проблеме быстрого износа твердотельных накопителей (SSD) при использовании их в качестве кэша. Основное внимание в работе уделяется исследованию ключевых характеристик существующих алгоритмов кэширования, таких как Hit Rate (отношение запросов к блокам, которые кэшированы к общему числу запросов), временная сложность работы алгоритма, объем записываемой на диск информации.  В статье произведено сравнение показателей этих параметров для некоторых, наиболее подходящих для использования с SSD кэшом алгоритмов. Кроме того, проведен теоретический анализ влияния этих параметров на срок службы SSD и производительность систем хранения данных (СХД).
В результате проведенного анализа установлено, какими характеристиками должен обладать новый алгоритм кэшированя, который позволит увеличить срок службы SSD накопителей в параллельном логструктурированном кэше без серьезного влияния на его производительность.
## Введение
Одним из главных изменений последних лет в сфере систем хранения данных (СХД) является активное использование твердотельных накопителей (SSD). На сегодняшний день технологии создания твердотельных накопителей позволяют достичь значительно более высокой скорости чтения и записи данных по сравнению с магнитными (HDD). Данная особенность позволяет использовать массивы SSD дисков в качестве кэша в системах хранения данных.
Однако, SSD и HDD накопители имеют различное внутреннее устройство. Так, SSD не имеют подвижных частей, что позволяет обрабатывать все виды запросов с одинаково высокой скоростью. Но главным недостатком данных накопителей является ограниченный срок службы. Существующие на данный момент алгоритмы кэширования, логика которых ведет к интенсивному использованию кэша, не предназначены для использования их с SSD накопителями. Это приводит к преждевременному их износу из-за большого числа операций перезаписи. Таким образом, необходимы новые алгоритмы, которые позволят уменьшить число операций перезаписи, не снизив при этом производительность СХД.
Цель данной работы - выработка критериев для разработки эффективного алгоритма вытеснения данных для параллельного логструктурированного SSD-кэша. Для достижения данной цели следует выполнить следующий ряд задач:
* Изучение существующих алгоритмов вытеснения;
* Сравнение алгоритмов по выбранным критериям (Hit Rate, временная сложность, объем записываемой на диск информации);
* Теоретический анализ влияния выбранных критериев на срок службы SSD и производительность СХД.
* Вывод о том, каким соотношениями значений выбранных критериев должен обладать новый алгоритм вытеснения для того, чтобы его можно было применять в параллельном логструктурированном кэше
## Обзор предметной области
Под алгоритмами кэширования (алгоритмами вытеснения или политиками вытеснения) понимается программная структура, которая способна управлять кэшем информации, хранимой в СХД. Когда кэш заполнен, алгоритм должен выбрать, что именно удалить из него, чтобы далее вести запись в кэш новой, более актуальной информации. При выборе алгоритмов для сравнения рассматривались как общие алгоритмы, созданные для работы с HDD и оперативной памятью, так и алгоритмы, которые создавались специально для работы с SSD кэшом. Кроме того, предпочтение отдавалось тем, в которых производилось сравнение с другими алгоритмам, так как это важно для приведения представленных в источниках результатов измерений по другим критериям в единую систему измерений для последующего сравнения. Ниже представлено краткое описание алгоритмов, выбранных для сравнения.
- LRFU
"Least Recently/Frequently Used" (Вытеснение давно/часто неиспользуемых). Общий алгоритм. Сочетает в себе слежение как за частотой, так и за давностью обращений путем вычисления величины CRF (Combined Recency and Frequency). Для этого хранится история последних нескольких обращений. CRF вычисляется суммированием значений какой-либо взвешивающей функции, вычисленной в точках, соответствующих временам обращений. [1]
- ARC
"Adaptive Rplacement Cache" (Кэш адаптивной замены). Общий алгоритм. Первый алгоритм, использующий адаптивность как одну из ключевых идей. Поддерживается два LRU(Least Recently Used) списка, первый из которых содержит блоки, обращение к которым произошло лишь единожды, второй - все остальные. У обоих списков имеется история недавних вытеснений из них и постоянно меняющийся целевой размер, к которому алгоритм пытается приблизить. [2]
- LARC
"Lazy Adaptive Rplacement Cache" (Ленивый кэш адаптивной замены). Алгоритм для работы с SSD. Поддерживаются две LRU-очереди. Одна из них (Main) имеет фиксированный размер и хранит страницы кэша. Вторая (Ghost) хранит только идентификаторы блоков и имеет переменный размер. Когда блок попадает в кэш, его идентификатор сохраняется в Ghost-очередь, а сам блок не записывается. Если происходит попадание (Hit) в блок из Ghost-очереди, то он сохраняется в кэш, а идентификатор из Ghost удаляется. Размер Ghost-очереди настраивается по ходу работы примерно так, как это сделано в ARC. [3]
- PLC
"Popular and Long-Term Cache" (Кэш популярных и долгосрочных данных). Алгоритм для работы с SSD. Поддерживает двухэтапный подход с ведением статистики, которая позволяет определить, к каким данным чаще всего происходит обращение (популярные), и при этом они не очень часто перезаписываются. Все хранящиеся данные разбиваются на три группы:
  * Популярные, редко перезаписываемые
  * Популярные, часто перезаписываемые
  * Непопулярные
  Как только некий блок попадает в кэш, для него начинается ведение еще двух счетчиков - "внутреннего приоритета" и "устаревания", на которых основано принятие решения о том, стоит ли заменять данный блок другим. [4]
- MQ
"Multi-Queue" (Несколько очередей). Общий алгоритм. Организует m LRU-очередей (m - параметр, допускающий настройку), а также очередь, содержащую историю недавно вытесненных блоков. Номер очереди, в которой должен содержаться блок, высчитывается применением монотонной функции к частоте запросов для этого блока. При попадании в кэш блок помещается в конец очереди с вычисленным номером. Вытесняется находящийся в голове непустой очереди с минимальным номером блок. Также блок, который пробыл в очереди некоторое определенное время, вытесняется в очередь с меньшим номером. [5]


Данные алгоритмы сравнивались по следующим критериям: Hit Rate (отношение запросов к блокам, которые кэшированы к общему числу запросов), временная сложность работы алгоритма, объем записываемой на диск информации.

Объем записываемой на диск информации является наиболее важным критерием, так как именно он отражает количество операций перезаписи. Чем меньше данное значение - тем больше алгоритм подходит для использования именно с SSD-накопителями. Так как SSD-накопители подвержены износу именно из-за частой перезаписи информации, то, чем меньший объем информации записывается на диск во время работы алгоритма, тем дольше прослужит SSD. В источниках приведены результаты для разных размеров кэша. Кроме того, данные даны в разных величинах - Гб и млн. блоков. Для сравнения были взяты данные для примерно одинакового размера кэша - 1200 тыс. блоков. Также данные приведены к единой системе сравнения - млн. блоков. Так в [3], откуда взяты данные по объему записываемой информации для LRFU, ARC, LARC и MQ, данные предоставлены в блоках, а в [4] - в ГБ и только для одного размера кэша - 1200 тыс. блоков. В обоих источниках было произведено сравнение с LRU, причем в [3] среди прочих имеются измерения для размера кэша в 1200 тыс. блоков. Таким образом, основываясь на значениях для LRU в 11 млн. блоков из [3] и в 48 ГБ из [4], а также значении для PLC в 3 ГБ из [4], можно установить, что размер записанных данных в блоках для PLC можно вычислить по формуле: РазмерБл_PLC = (РазмерГб_PLC * РазмерБл_LRU) / (РазмерГб_LRU).

Hit Rate должен быть как можно более высоким, так как именно он определяет эффективность алгоритма как кэширующего. При малом значении Hit Rate в кэше может храниться мало данных, к которым может потребоваться доступ, что не принесет пользы - кэш не будет обеспечивать высокую скорость доступа к данным.

Временная сложность алгоритма определяет, насколько затратным по времени является алгоритм. В качестве параметра n - число блоков в кэше. Определяется как функция от числа блоков в кэше. При вычислении учитывается слагаемое самого высокого порядка и не учитываются константные множители (коэффициенты). Временная сложность оценивается путём подсчёта числа элементарных операций, осуществляемых алгоритмом. Время исполнения одной такой операции при этом берётся константой, то есть, асимптотически оценивается как O(1). В таких обозначениях полное время исполнения и число элементарных операций, выполненных алгоритмом, отличаются максимум на постоянный множитель, который не учитывается. Чем менее затратным по времени является алгоритм, тем меньшая задержка происходит при записи или перезаписи данных. Как следствие, уменьшается падение производительности системы в целом.

Результаты сравнения выбранных алгоритмов по вышеописанным критериям представлены в табл. 1.

Как можно отметить, объем записанной информации разнится, но у алгоритмов, которые разрабатывались для работы с SSD кэшом (LARC, PLC) это значение в разы меньше, что подтверждает эффективность данных алгоритмов для работы именно с SSD. 

Процент попадания в кэш также находится на высоком уровне. Следует отметить, что у подавляющего большинства алгоритмов с ростом размера кэша растет и Hit Rate, но при этом на представленных в [1]-[3] графиках при самых больших размерах кэша разница в Hit Rate была минимальна даже по сравнению с алгоритмом LRU, с которым сравнивалось подавляющее большинство алгоритмов. Сложность почти всех алгоритмов константная, так как алгоритмы в большинстве своем основаны на LRU, сложность которого О(1).

## Выбор метода решения
На основе проведенного исследования аналогов можно отметить, что значения Hit Rate является на данный момент величиной, которая стремится примерно к одному и тому же значению при увеличении размера кэша. Также сложность по времени большинства алгоритмов не превышает константную. При этом с уменьшением числа записываемых в кэш данных при неизменном значением Hit Rate сложность алгоритма будет расти.

Отличие параллельного логструктурированного кэша от обычного состоит в том, что он работает параллельно с оперативной памятью, в то время как обычный кэш работает последовательно с оперативной памятью, то есть по сути является ее продолжением. 

Учитывая само устройство кэша, можно отметить, что при правильном использовании его совместно с оперативной памятью, можно добиться того, что даже применение алгоритма LRU даст в результате большое значение Hit Rate. Например, этого можно добиться разбиением доступного пространства кэша на одинаковые по объему части (дескрипторы), а информацию по каждому из них хранить в оперативной памяти. Это также позволит увеличить число записываемой на диск полезной информации. Также в оперативной памяти можно хранить дескрипторы, к которым было только одно обращение (создать некое подобие ghost-очереди).

В итоге, чтобы быть эффективным, разработанное решение в первую очередь должно обеспечивать возможность как можно более низкое число записываемых на диск данных. Сложность алгоритма по времени при этом не должна быть высокой, чтобы не возникало длительных задержек во время работы с записью или перезаписью. Следует отметить, что сложность алгоритмов, которые основаны на LRU, была константной или близкой к ней. 

Кроме того, нужен эффективный способ отбора кандидатов на вытеснение из кэша, который будет эффективно использовать тот факт, что кэш работает параллельно с оперативной памятью. При этом во внимание должны приниматься такие факторы, как востребованность блока (как часто к нему обращаются) и время нахождения блока в кэше без обращений. Алгоритм должен поддерживать в кэше как можно больше блоков, к которым часто обращаются.

## Заключение
В статье представлен анализ существующих алгоритмов кэширования - LRFU, ARC, LARC, PLC и MQ. Анализ производился по критериям Hit Rate, временная сложность работы алгоритма, объем записываемой на диск информации. В результате анализа было установлено, что непосредственно способ выбора того, какую информацию удалить из кэша (выбор алгоритма), не оказывает сильного влияния на Hit Rate, особенно при большом размере кэша. Кроме того, было отмечено, что сложность почти всех исследованных алгоритмов константная, так как они используют в качестве основы алгоритм LRU. Было также установлено, что у тех алгоритмов, которые разрабатывались непосредственно для работы с SSD кэшом, объем записываемой на диск информации в разы ниже, чем у других алгоритмов.

С учетом анализа существующих алгоритмов и структуры параллельного SSD кэша было установлено, что главным критерием, на который следует обращать внимание при разработке алгоритма - это объем записываемой на диск информации. Применительно к параллельному кэшу было также выдвинуто предположение, что его принцип работы может позволить достичь больших значений Hit Rate за счет правильного использования оперативной памяти даже при использовании менее эффективных алгоритмов. Сложность алгоритма по времени при этом не должна быть высокой для того, чтобы не было задержек во время выявления данных, которые необходимо вытеснить из кэша.

В дальнейшем результатами данного анализа следует пользоваться при создании нового алгоритма вытеснения, что позволит снизить его влияние на срок службы SSD накопителей, не оказывая при этом сильного влияния на производительность и не теряя преимущества, из-за которых используется кэш - быстрый доступ к часто используемым данным.