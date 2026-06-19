### Z-Fast Trie (Обзор Структуры)

Z-Fast Trie — это компактная структура данных, предложенная Belazzougui et al. [[ZFastTrie & MonotoneMinimalPerfectHashing]](https://vigna.di.unimi.it/ftp/papers/MonotoneMinimalPerfectHashing.pdf), которая обеспечивает эффективную навигацию по множеству
строк. Она играет центральную роль в задачах, требующих малого потребления памяти (Little Space), таких как индексация больших коллекций
строк или построение минимальных совершенных хеш-функций.

## Применения в системе

В контексте нашей системы (Approximate Range Emptiness), Z-Fast Trie используется в двух ключевых компонентах:

- **Hollow Z-Fast Trie**: Как навигатор для определения "узла выхода" (exit node)
- **MMPH Distributor**: Как распределитель ключей по ведрам для сохранения монотонности

## Определения задач

### 1. Prefix Search (Поиск по Префиксу)

> *"...given a collection of strings, find all the strings that start with a given prefix."*  
> [[Fast prefix search in little space, with applications]](https://arxiv.org/abs/1804.04720), Sec 1

- **Вход**: Строка-запрос $p$
- **Выход**: Идентификация всех строк из множества $S$, имеющих префикс $p$
- **В контексте Trie**: Это эквивалентно нахождению узла $u$, чей путь (path) совпадает с $p$

### 2. Weak Prefix Search (Слабый Поиск по Префиксу)

> *"...given a prefix $p$ of some string in $S$, the weak prefix search problem requires... to return the range of strings of $S$ having $p$
as prefix; this set is returned as the interval of integers that are the ranks (in lexicographic order)..."*  
> [[Fast prefix search in little space, with applications]](https://arxiv.org/abs/1804.04720), Sec 2

- **Вход**: Префикс $p$ некоторой строки из множества $S$
- **Выход**: Диапазон рангов $[i, j)$ - интервал целых чисел, представляющих ранги (в лексикографическом порядке) всех строк, имеющих $p$ в
  качестве префикса

**Особенность**: Если $p$ не является префиксом ни одной строки из $S$, поведение структуры не определено (или она возвращает мусор).
Однако, если $p$ валиден, она обязана вернуть корректный диапазон рангов $[i, j)$.

**Роль Z-Fast Trie**: Сама структура Z-Fast Trie решает навигационную часть задачи — находит "узел выхода" (exit node). Превращение узла в
диапазон рангов выполняет компонент Range Locator.

### 3. Predecessor Search (Поиск Предшественника)

> *"...return the position of the largest key not greater than $x$."*  
> [[ZFastTrie & MonotoneMinimalPerfectHashing]](https://vigna.di.unimi.it/ftp/papers/MonotoneMinimalPerfectHashing.pdf), Sec 1

**Применение**: Используется внутри MMPH (Distributor). Z-Fast Trie позволяет найти, в какое ведро (bucket) попадает ключ, даже если точного
совпадения нет.

## Варианты реализации Z-Fast Trie

Существует два основных подхода к реализации Z-Fast Trie, каждый с собственными преимуществами и ограничениями.

### Deterministic Z-Fast Trie

Implementation was adopted to Go from [C++](https://gitlab.com/habatakitai/ctriepp)


**Подход**: Хранит полные строки и их длины в узлах дерева.

**Характеристики**:

- **Простота реализации**: Прямолинейная логика без вероятностных элементов
- **Детерминированность**: Гарантированные результаты без ложных срабатываний
- **Отладка**: Легко отслеживать и проверять состояние структуры
- **Пространственная сложность**: $O(n \cdot l)$ байт, где $l$ — средняя длина строки

**Применение**: Идеально для прототипирования и случаев, где простота важнее экстремальной оптимизации памяти.

**Текущее состояние**: Используется в нашем прототипе для отладки и тестирования логики.

### Probabilistic Z-Fast Trie

Согласно статье Monotone Minimal Perfect Hashing (Section 4.1 "A probabilistic trie"), для достижения теоретически оптимального потребления
памяти используется Probabilistic Z-Fast Trie.

**Подход**: Вместо хранения полных строк или длин в узлах дерева, структура хранит компактные сигнатуры (signatures/fingerprints) путей.

**Характеристики**:

- **Экстремальная оптимизация памяти**: $O(m \log \log w)$ бит (где $m$ — число ключей, $w$ — длина слова)
- **Скорость**: Оптимизирован для операций на машинном слове
- **2-fattest кодирование**: Использует математические свойства для эффективного представления префиксов

**Риск**: Существует малая вероятность ложноположительного срабатывания (false positive) при сравнении сигнатур.

**Решение**: В контексте MMPH ошибка устраняется проверкой внутри ведра (Bucketing) или использованием относительного кодирования (Relative
Trie), где мы уверены, что запрос принадлежит множеству $S$.

> *"To describe the probabilistic trie, we need some simple properties of 2-fattest numbers... A probabilistic z-fast trie is given by a
function T... that maps the prefix of p... to... a signature of p..."*  
> [[ZFastTrie & MonotoneMinimalPerfectHashing]](https://vigna.di.unimi.it/ftp/papers/MonotoneMinimalPerfectHashing.pdf), Sec 4.1

**Целевое применение**: Критически важно для MMPH в продакшене, где пространственная эффективность является приоритетом.

## API структуры

Интерфейс Z-Fast Trie должен быть универсальным, чтобы обслуживать нужды как Hollow Trie (навигация), так и MMPH (распределение).
Вот так может выглядеть примерный API который мы хотим реализовать

```go
type PrefixSearchResult[V] struct {
ExitNode    V     // "Узел выхода" (exit node) для передачи в Range Locator
MatchLength int   // Длина совпавшего префикса
IsExactMatch bool // Точное ли это совпадение с существующим ключом
}

type PredecessorSearchResult[V] struct {
BucketID    V   // ID ведра для размещения ключа
Position    int // Позиция в лексикографическом порядке
}

type RankRange struct {
Start int // Начальный ранг (включительно)
End   int // Конечный ранг (исключительно) [Start, End)
}
```

### Основной интерфейс

```go
interface ZFastTrie[V] {
    Build(keys map[string]V)
    PrefixSearch(query string) *PrefixSearchResult[V]
    PredecessorSearch(query string) PredecessorSearchResult[V]
}

// Range Locator - отдельный компонент, преобразующий exit node в диапазон рангов
interface RangeLocator[V] {
    NodeToRankRange(exitNode V) RankRange
}
```

### Future structures built on Trie

```go
// Для Hollow Trie (навигация)
type HollowTrieNavigator[V] struct {
zfastTrie     ZFastTrie[V]
rangeLocator  RangeLocator[V]
}

func (nav *HollowTrieNavigator[V]) WeakPrefixSearch(prefix string) RankRange {
// 1. Z-Fast Trie находит exit node
result := nav.zfastTrie.PrefixSearch(prefix)
// 2. Range Locator преобразует узел в диапазон рангов
return nav.rangeLocator.NodeToRankRange(result.ExitNode)
}

// Для MMPH Distributor (распределение по ведрам)
func (distributor *MMPHDistributor[V]) FindBucket(key string) int {
result := distributor.zfastTrie.PredecessorSearch(key)
return int(result.BucketID)
}
```

## Применения в системе

### A. Навигация в Hollow Trie (Уровень 1)

**Задача**: Для префикса $p$ найти узел $u$ в сжатом дереве.

**Использование**: Вызывается `weakSearch(p)`.

**Результат**: Z-Fast Trie возвращает "ручку" (handle) или ID узла. Далее этот ID передается в Range Locator для получения диапазона
рангов $[L, R]$.

### B. Распределитель в MMPH (Range Locator)

**Задача**: Для ключа $x$ (имени узла) найти индекс ведра $B_i$.

**Использование**: Вызывается `weakSearch(x)` (или Predecessor Search).

**Контекст**: MMPH разбивает ключи на ведра. Z-Fast Trie (в идеале Probabilistic) хранит "разделители" ведер. Поиск позволяет за $O(1)$
или $O(\log L)$ определить, в каком ведре лежит ключ, чтобы затем локально вычислить его ранг.

## Сравнительная характеристика реализаций

### Производительность и асимптотика

| Характеристика       | Deterministic Z-Fast Trie | Probabilistic Z-Fast Trie |
|:---------------------|:--------------------------|:--------------------------|
| **Время поиска**     | $O(\log l)$               | $O(p/w + \log w)$         |
| **Память**           | $O(n \cdot l)$ байт       | $O(n \log \log w)$ бит    |
| **Время построения** | $O(n \cdot l)$            | $O(n)$                    |

### Теоретические гарантии

**Probabilistic Z-Fast Trie** (согласно статье):
> *"The structure requires $O(m(\log w + \log(1/\epsilon)))$ bits of space and has query time $O(\log w)$."*  
> [[ZFastTrie & MonotoneMinimalPerfectHashing]](https://vigna.di.unimi.it/ftp/papers/MonotoneMinimalPerfectHashing.pdf), Theorem 4.1

**Где**:

- $m$ — количество ключей
- $w$ — размер машинного слова
- $l$ — средняя длина строки
- $\epsilon$ — допустимая вероятность ошибки

### Стратегия миграции

1. **Фаза 1**: Реализация Deterministic версии для валидации алгоритмов
2. **Фаза 2**: Параллельная реализация Probabilistic версии
3. **Фаза 3**: A/B тестирование производительности
4. **Фаза 4**: Переход на Probabilistic для продакшена
